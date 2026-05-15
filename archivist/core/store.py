"""
FAISS + SQLite unified storage for Archivist-AI.

Architecture
------------
- FAISS IndexFlatIP  : stores L2-normalised float32 vectors.
                       Inner product == cosine similarity for unit vectors.
- SQLite (WAL mode)  : stores per-image metadata — file path, SHA-256 hash,
                       tags, timestamps.

Why two layers?
- FAISS is incredibly fast for ANN search but has no notion of metadata,
  deletion, or filtering.
- SQLite fills those gaps: we can filter out "deleted" rows (files that were
  removed from disk) without rebuilding the FAISS index, store arbitrary tags,
  and query by path/hash efficiently.

Deletion model
--------------
FAISS IndexFlatIP does not support removing individual vectors. Deleted entries
are removed from SQLite so they never surface in search results. The FAISS
index grows monotonically; call `rebuild()` periodically to reclaim space.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class ImageRecord:
    """Full metadata for a single indexed image."""
    faiss_id: int
    file_hash: str
    file_path: str
    filename: str
    file_size: int
    tags: list[str] = field(default_factory=list)
    indexed_at: str = ""


@dataclass
class SearchResult:
    """A single ranked search result."""
    rank: int
    file_path: Path
    filename: str
    similarity: float           # Cosine similarity in [−1, 1]; practically [0, 1]
    tags: list[str]
    faiss_id: int
    date_taken: str | None = None  # ISO-8601 string e.g. "2023-06-15T14:30:00"


# ── Store ─────────────────────────────────────────────────────────────────────


class ArchivistStore:
    """
    Unified FAISS + SQLite store.

    Usage (context manager — preferred):
    ─────────────────────────────────────
        with ArchivistStore(index_dir) as store:
            store.add(...)
            results = store.search(query_emb, k=20)

    Manual lifecycle:
    ─────────────────
        store = ArchivistStore(index_dir)
        store.open()
        ...
        store.close()   # flushes FAISS to disk
    """

    FAISS_FILE = "vectors.index"
    DB_FILE = "metadata.db"

    def __init__(self, index_dir: Path | str, embedding_dim: int = 512) -> None:
        self.index_dir = Path(index_dir)
        self.embedding_dim = embedding_dim
        self._index: Optional[faiss.IndexFlatIP] = None
        self._db: Optional[sqlite3.Connection] = None
        self._dirty = False  # True when FAISS has unsaved changes

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def faiss_path(self) -> Path:
        return self.index_dir / self.FAISS_FILE

    @property
    def db_path(self) -> Path:
        return self.index_dir / self.DB_FILE

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> "ArchivistStore":
        """Open (or create) the store. Returns self for chaining."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._open_db()
        self._open_faiss()
        logger.debug(
            f"Store opened at {self.index_dir}  "
            f"({self.count()} images, {self._index.ntotal} vectors)"
        )
        return self

    def close(self) -> None:
        """Flush FAISS to disk and close SQLite."""
        self.save()
        if self._db:
            self._db.close()
            self._db = None

    def save(self) -> None:
        """Persist the FAISS index to disk (no-op if nothing changed)."""
        if self._index is not None and self._dirty:
            faiss.write_index(self._index, str(self.faiss_path))
            self._dirty = False
            logger.debug(f"FAISS index saved  ({self._index.ntotal} vectors)")

    def __enter__(self) -> "ArchivistStore":
        return self.open()

    def __exit__(self, *_) -> None:
        self.close()

    # ── DB schema ─────────────────────────────────────────────────────────────

    def _open_db(self) -> None:
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS images (
                faiss_id    INTEGER PRIMARY KEY,
                file_hash   TEXT    NOT NULL UNIQUE,
                file_path   TEXT    NOT NULL,
                filename    TEXT    NOT NULL,
                file_size   INTEGER NOT NULL DEFAULT 0,
                indexed_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS tags (
                image_id    INTEGER NOT NULL
                                REFERENCES images(faiss_id) ON DELETE CASCADE,
                tag         TEXT    NOT NULL,
                confidence  REAL    NOT NULL DEFAULT 1.0,
                PRIMARY KEY (image_id, tag)
            );

            CREATE INDEX IF NOT EXISTS idx_images_hash ON images(file_hash);
            CREATE INDEX IF NOT EXISTS idx_images_path ON images(file_path);
            CREATE INDEX IF NOT EXISTS idx_tags_tag    ON tags(tag);
        """)
        self._db.commit()

        # ── Safe migration: add date_taken to existing databases ──────────────
        try:
            self._db.execute("ALTER TABLE images ADD COLUMN date_taken TEXT")
            self._db.commit()
            logger.debug("Migrated: added date_taken column")
        except sqlite3.OperationalError:
            pass  # Column already exists — nothing to do

        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_images_date ON images(date_taken)"
        )
        self._db.commit()

        # ── Backfill date_taken from file mtime for pre-migration records ─────
        # Uses only stat() — no image loading — so it's fast even for large indexes.
        stale = self._db.execute(
            "SELECT faiss_id, file_path FROM images WHERE date_taken IS NULL"
        ).fetchall()
        if stale:
            from datetime import datetime
            updates: list[tuple[str, int]] = []
            for row in stale:
                try:
                    mtime = Path(row["file_path"]).stat().st_mtime
                    dt = datetime.fromtimestamp(mtime).strftime("%Y-%m-%dT%H:%M:%S")
                    updates.append((dt, row["faiss_id"]))
                except OSError:
                    pass
            if updates:
                self._db.executemany(
                    "UPDATE images SET date_taken = ? WHERE faiss_id = ?", updates
                )
                self._db.commit()
                logger.debug(f"Backfilled date_taken for {len(updates)} existing records")

    def _open_faiss(self) -> None:
        if self.faiss_path.exists():
            self._index = faiss.read_index(str(self.faiss_path))
            logger.debug(f"Loaded FAISS index  ({self._index.ntotal} vectors)")
            if self._index.d != self.embedding_dim:
                raise RuntimeError(
                    f"\n\n"
                    f"  ❌  Embedding dimension mismatch!\n\n"
                    f"  Existing index was built with {self._index.d}-dim vectors\n"
                    f"  (probably CLIP), but the current model produces\n"
                    f"  {self.embedding_dim}-dim vectors (SigLIP).\n\n"
                    f"  To switch models, delete the old index and re-index:\n\n"
                    f"    1. Delete:    {self.index_dir}\n"
                    f"    2. Re-index:  archivist index <your-photo-folder>\n"
                )
        else:
            self._index = faiss.IndexFlatIP(self.embedding_dim)
            logger.debug("Created new FAISS IndexFlatIP")

    # ── Write operations ──────────────────────────────────────────────────────

    def add(
        self,
        file_hash: str,
        file_path: Path | str,
        embedding: np.ndarray,
        tags: list[tuple[str, float]] | None = None,
        date_taken: str | None = None,
    ) -> int:
        """
        Add a single image to the store.

        Returns the FAISS ID assigned. If the hash already exists, returns
        the existing FAISS ID without re-adding.
        """
        existing = self._db.execute(
            "SELECT faiss_id FROM images WHERE file_hash = ?", (file_hash,)
        ).fetchone()
        if existing:
            return existing["faiss_id"]

        faiss_id = self._index.ntotal
        vec = embedding.reshape(1, -1).astype(np.float32)
        self._index.add(vec)
        self._dirty = True

        file_path = Path(file_path)
        try:
            fsize = file_path.stat().st_size
        except OSError:
            fsize = 0

        self._db.execute(
            """INSERT INTO images (faiss_id, file_hash, file_path, filename, file_size, date_taken)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (faiss_id, file_hash, str(file_path), file_path.name, fsize, date_taken),
        )
        if tags:
            self._db.executemany(
                "INSERT OR IGNORE INTO tags (image_id, tag, confidence) VALUES (?, ?, ?)",
                [(faiss_id, tag, conf) for tag, conf in tags],
            )
        self._db.commit()
        return faiss_id

    def add_batch(
        self,
        records: list[dict],
        embeddings: np.ndarray,
    ) -> list[int]:
        """
        Batch-add images. Much faster than looping add().

        Parameters
        ----------
        records    : list of dicts with keys: file_hash, file_path, tags (optional)
        embeddings : float32 array (N, dim) aligned with records

        Returns
        -------
        list of FAISS IDs (one per record; existing records return their old ID)
        """
        assert len(records) == len(embeddings), "records and embeddings must be the same length"

        start_id = self._index.ntotal
        faiss_ids: list[int] = []
        new_vecs: list[np.ndarray] = []
        new_db_rows: list[tuple] = []
        new_tag_rows: list[tuple] = []
        seen_in_batch: set[str] = set()   # guard against within-batch hash collisions

        for record, emb in zip(records, embeddings):
            file_hash = record["file_hash"]

            # Skip if already persisted in a previous batch
            existing = self._db.execute(
                "SELECT faiss_id FROM images WHERE file_hash = ?",
                (file_hash,),
            ).fetchone()

            if existing:
                faiss_ids.append(existing["faiss_id"])
                continue

            # Skip true duplicates within the current batch (same content, different names)
            if file_hash in seen_in_batch:
                logger.debug(f"Skipping within-batch duplicate hash: {record.get('file_path', '?')}")
                continue
            seen_in_batch.add(file_hash)

            faiss_id = start_id + len(new_vecs)
            new_vecs.append(emb)

            fp = Path(record["file_path"])
            try:
                fsize = fp.stat().st_size
            except OSError:
                fsize = 0

            new_db_rows.append(
                (faiss_id, record["file_hash"], str(fp), fp.name, fsize,
                 record.get("date_taken"))
            )

            for tag, conf in record.get("tags", []):
                new_tag_rows.append((faiss_id, tag, conf))

            faiss_ids.append(faiss_id)

        if new_vecs:
            mat = np.array(new_vecs, dtype=np.float32)
            self._index.add(mat)
            self._dirty = True

            self._db.executemany(
                """INSERT INTO images (faiss_id, file_hash, file_path, filename, file_size, date_taken)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                new_db_rows,
            )
            if new_tag_rows:
                self._db.executemany(
                    "INSERT OR IGNORE INTO tags (image_id, tag, confidence) VALUES (?, ?, ?)",
                    new_tag_rows,
                )
            self._db.commit()

        return faiss_ids

    def update_tags(
        self,
        faiss_id: int,
        tags: list[tuple[str, float]],
        replace: bool = False,
    ) -> None:
        """Add or replace tags for a given image."""
        if replace:
            self._db.execute("DELETE FROM tags WHERE image_id = ?", (faiss_id,))
        self._db.executemany(
            "INSERT OR REPLACE INTO tags (image_id, tag, confidence) VALUES (?, ?, ?)",
            [(faiss_id, tag, conf) for tag, conf in tags],
        )
        self._db.commit()

    def remove_by_hash(self, file_hash: str) -> None:
        """Mark an image as deleted (removes from SQLite; FAISS grows monotonically)."""
        self._db.execute("DELETE FROM images WHERE file_hash = ?", (file_hash,))
        self._db.commit()

    def remove_missing_files(self) -> int:
        """
        Remove SQLite records for images whose files no longer exist on disk.

        Returns the number of stale records removed.
        """
        rows = self._db.execute("SELECT faiss_id, file_path FROM images").fetchall()
        stale_ids = [row["faiss_id"] for row in rows if not Path(row["file_path"]).exists()]

        if stale_ids:
            self._db.executemany(
                "DELETE FROM images WHERE faiss_id = ?",
                [(fid,) for fid in stale_ids],
            )
            self._db.commit()
            logger.info(f"Removed {len(stale_ids)} stale records")

        return len(stale_ids)

    # ── Read operations ───────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 20,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[SearchResult]:
        """
        Find the k most similar images to query_embedding.

        Uses FAISS inner product (== cosine sim for L2-normed vectors).
        Records deleted from SQLite are silently skipped.

        Parameters
        ----------
        date_from : ISO date string "YYYY-MM-DD" — only return images on or after this date.
        date_to   : ISO date string "YYYY-MM-DD" — only return images on or before this date.
        """
        if self._index.ntotal == 0:
            return []

        has_date_filter = bool(date_from or date_to)

        # When date filtering, ask FAISS for more candidates so we can fill k
        # results after the filter removes some.
        k_query = min(k * 5 if has_date_filter else k, self._index.ntotal)
        vec = query_embedding.reshape(1, -1).astype(np.float32)
        scores, ids = self._index.search(vec, k_query)

        results: list[SearchResult] = []
        rank = 1
        for faiss_id, score in zip(ids[0], scores[0]):
            if faiss_id < 0:
                continue
            if len(results) >= k:
                break

            row = self._db.execute(
                "SELECT file_path, filename, date_taken FROM images WHERE faiss_id = ?",
                (int(faiss_id),),
            ).fetchone()

            if not row:
                continue  # Soft-deleted; skip

            # Apply date filter — images with no stored date are excluded when
            # a filter is active (they were indexed before this feature existed
            # and mtime backfill hasn't run yet for them).
            if has_date_filter:
                dt = row["date_taken"]
                if not dt:
                    continue
                # Compare ISO strings lexicographically — works because the
                # format is "YYYY-MM-DD..." which sorts the same as a real date.
                if date_from and dt[:10] < date_from[:10]:
                    continue
                if date_to and dt[:10] > date_to[:10]:
                    continue

            tag_rows = self._db.execute(
                "SELECT tag FROM tags WHERE image_id = ? ORDER BY confidence DESC",
                (int(faiss_id),),
            ).fetchall()
            tags = [r["tag"] for r in tag_rows]

            results.append(SearchResult(
                rank=rank,
                file_path=Path(row["file_path"]),
                filename=row["filename"],
                similarity=float(score),
                tags=tags,
                faiss_id=int(faiss_id),
                date_taken=row["date_taken"],
            ))
            rank += 1

        return results

    def get_date_range(self) -> tuple[str | None, str | None]:
        """Return (earliest_date, latest_date) across all indexed images."""
        row = self._db.execute(
            "SELECT MIN(date_taken) AS min_d, MAX(date_taken) AS max_d "
            "FROM images WHERE date_taken IS NOT NULL"
        ).fetchone()
        if row:
            return row["min_d"], row["max_d"]
        return None, None

    def get_all_metadata(self) -> list[dict]:
        """
        Return metadata for every non-deleted image.

        Used by the UMAP browser to align vectors with file paths.
        """
        rows = self._db.execute(
            "SELECT faiss_id, file_path, filename, date_taken "
            "FROM images ORDER BY faiss_id"
        ).fetchall()
        return [dict(row) for row in rows]

    def get_all_hashes(self) -> set[str]:
        """Return set of all indexed file hashes (for incremental indexing)."""
        rows = self._db.execute("SELECT file_hash FROM images").fetchall()
        return {row["file_hash"] for row in rows}

    def get_all_vectors(self) -> tuple[np.ndarray, list[int]]:
        """
        Return all stored vectors and their FAISS IDs.

        Used by DuplicateFinder. Retrieves raw vectors from FAISS xb storage.
        Note: includes soft-deleted vectors (check against SQLite if needed).
        """
        n = self._index.ntotal
        if n == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32), []

        # Reconstruct all stored vectors using the FAISS reconstruct_n API
        # (IndexFlatIP does not expose an .xb attribute in all FAISS builds)
        all_vecs = np.zeros((n, self.embedding_dim), dtype=np.float32)
        self._index.reconstruct_n(0, n, all_vecs)
        return all_vecs, list(range(n))

    def get_record(self, faiss_id: int) -> ImageRecord | None:
        """Fetch full metadata for a single image by FAISS ID."""
        row = self._db.execute(
            "SELECT * FROM images WHERE faiss_id = ?", (faiss_id,)
        ).fetchone()
        if not row:
            return None

        tags = [r["tag"] for r in self._db.execute(
            "SELECT tag FROM tags WHERE image_id = ? ORDER BY confidence DESC",
            (faiss_id,),
        ).fetchall()]

        return ImageRecord(
            faiss_id=row["faiss_id"],
            file_hash=row["file_hash"],
            file_path=row["file_path"],
            filename=row["filename"],
            file_size=row["file_size"],
            tags=tags,
            indexed_at=row["indexed_at"],
        )

    def get_faiss_id_by_path(self, file_path: Path | str) -> int | None:
        """Look up FAISS ID from a file path."""
        row = self._db.execute(
            "SELECT faiss_id FROM images WHERE file_path = ?", (str(file_path),)
        ).fetchone()
        return row["faiss_id"] if row else None

    def count(self) -> int:
        """Number of images in SQLite (the authoritative count, excluding soft-deletes)."""
        if not self._db:
            return 0
        return self._db.execute("SELECT COUNT(*) FROM images").fetchone()[0]

    def stats(self) -> dict:
        """Return a summary of store statistics."""
        return {
            "total_indexed": self.count(),
            "faiss_vectors": self._index.ntotal if self._index else 0,
            "embedding_dim": self.embedding_dim,
            "index_dir": str(self.index_dir),
            "db_size_mb": round(self.db_path.stat().st_size / 1e6, 2) if self.db_path.exists() else 0,
            "index_size_mb": round(self.faiss_path.stat().st_size / 1e6, 2) if self.faiss_path.exists() else 0,
        }

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"ArchivistStore("
            f"index_dir={self.index_dir!r}, "
            f"indexed={self.count()}, "
            f"faiss_vectors={self._index.ntotal if self._index else 0})"
        )
