"""
Image directory indexer for Archivist-AI.

Key design decisions
────────────────────
- Incremental indexing: SHA-256 hashes are checked against the store before
  embedding. Only new / changed files are processed, making subsequent runs
  near-instant for large collections.
- Batch embedding: images are embedded in configurable batches (default 32)
  to balance throughput and peak memory usage on CPU.
- Graceful errors: unreadable files are counted and reported, never crash.
- Rich progress bar: shows real-time speed (images/sec) and ETA.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .embedder import BaseEmbedder
from .store import ArchivistStore

logger = logging.getLogger(__name__)


def _extract_date(path: Path) -> str | None:
    """
    Return an ISO-8601 date-time string for *path*.

    Priority:
    1. EXIF DateTimeOriginal / DateTimeDigitized / DateTime
    2. File modification time (mtime) as fallback

    Never raises — returns None if everything fails.
    """
    # ── 1. Try EXIF ───────────────────────────────────────────────────────────
    try:
        from PIL import Image as _PILImage
        from PIL.ExifTags import TAGS as _TAGS

        with _PILImage.open(path) as _img:
            _exif = _img._getexif()  # type: ignore[attr-defined]
            if _exif:
                for _tag_id, _value in _exif.items():
                    if _TAGS.get(_tag_id) in (
                        "DateTimeOriginal", "DateTimeDigitized", "DateTime"
                    ):
                        if isinstance(_value, str) and len(_value) >= 10:
                            # EXIF format: "2023:01:15 14:30:00"
                            # Normalise to:  "2023-01-15T14:30:00"
                            return (
                                _value[:10].replace(":", "-")
                                + "T"
                                + _value[11:19]
                            )
    except Exception:
        pass

    # ── 2. Fall back to mtime ─────────────────────────────────────────────────
    try:
        from datetime import datetime
        mtime = path.stat().st_mtime
        return datetime.fromtimestamp(mtime).strftime("%Y-%m-%dT%H:%M:%S")
    except Exception:
        return None


# Extensions we attempt to embed (Pillow-supported + HEIC via pillow-heif if installed)
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".gif", ".bmp",
    ".webp", ".tiff", ".tif", ".heic", ".heif",
})


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class IndexResult:
    """Summary of a completed indexing run."""
    total_found: int = 0
    newly_indexed: int = 0
    skipped_existing: int = 0
    skipped_errors: int = 0
    directories_scanned: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Fraction of found files that were successfully processed."""
        if self.total_found == 0:
            return 1.0
        return (self.newly_indexed + self.skipped_existing) / self.total_found

    def __str__(self) -> str:
        return (
            f"IndexResult("
            f"found={self.total_found}, "
            f"new={self.newly_indexed}, "
            f"cached={self.skipped_existing}, "
            f"errors={self.skipped_errors})"
        )


# ── Indexer ───────────────────────────────────────────────────────────────────


class ArchivistIndexer:
    """
    Walks directories, extracts CLIP embeddings, and persists them to the store.

    Incremental by design: re-running on the same folder only processes files
    that are new or whose contents have changed (detected via SHA-256 hash).

    Example
    -------
    >>> indexer = ArchivistIndexer(store, embedder)
    >>> result = indexer.index_directory(Path("~/Photos"))
    >>> print(f"Indexed {result.newly_indexed} new images.")
    """

    def __init__(
        self,
        store: ArchivistStore,
        embedder: BaseEmbedder,
        batch_size: int = 32,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.batch_size = batch_size

    # ── Public API ────────────────────────────────────────────────────────────

    def index_directory(
        self,
        directory: Path | str,
        recursive: bool = True,
        show_progress: bool = True,
    ) -> IndexResult:
        """
        Index all supported images under *directory*.

        Parameters
        ----------
        directory      : Root directory to scan.
        recursive      : Whether to descend into subdirectories.
        show_progress  : Show a Rich progress bar (set False in tests / watch mode).

        Returns
        -------
        IndexResult with counts of newly indexed, skipped, and errored files.
        """
        directory = Path(directory).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        result = IndexResult(directories_scanned=[str(directory)])

        # 1. Discover all image files
        image_paths = self._collect_images(directory, recursive)
        result.total_found = len(image_paths)

        if not image_paths:
            logger.info(f"No supported images found in {directory}")
            return result

        # 2. Filter to only files not yet in the store
        existing_hashes = self.store.get_all_hashes()
        new_paths, result.skipped_existing = self._filter_new_files(
            image_paths, existing_hashes
        )

        logger.info(
            f"Found {result.total_found} images  |  "
            f"{result.skipped_existing} already indexed  |  "
            f"{len(new_paths)} to embed"
        )

        if not new_paths:
            return result

        # 3. Embed and store in batches
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            disable=not show_progress,
        ) as progress:
            task = progress.add_task(
                f"Embedding {len(new_paths)} images...", total=len(new_paths)
            )

            for i in range(0, len(new_paths), self.batch_size):
                batch = new_paths[i : i + self.batch_size]
                batch_result = self._index_batch(batch)
                result.newly_indexed += batch_result["indexed"]
                result.skipped_errors += batch_result["errors"]
                progress.advance(task, len(batch))

        # 4. Flush FAISS to disk
        self.store.save()
        return result

    def index_multiple_directories(
        self,
        directories: list[Path | str],
        recursive: bool = True,
        show_progress: bool = True,
    ) -> IndexResult:
        """Index multiple directories, merging results into a single IndexResult."""
        combined = IndexResult()
        for directory in directories:
            result = self.index_directory(
                directory, recursive=recursive, show_progress=show_progress
            )
            combined.total_found += result.total_found
            combined.newly_indexed += result.newly_indexed
            combined.skipped_existing += result.skipped_existing
            combined.skipped_errors += result.skipped_errors
            combined.directories_scanned.extend(result.directories_scanned)
        return combined

    # ── Internals ─────────────────────────────────────────────────────────────

    def _collect_images(self, directory: Path, recursive: bool) -> list[Path]:
        """Collect all supported image files in directory order."""
        glob_pattern = "**/*" if recursive else "*"
        paths = [
            p for p in directory.glob(glob_pattern)
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        return sorted(paths)

    def _filter_new_files(
        self,
        image_paths: list[Path],
        existing_hashes: set[str],
    ) -> tuple[list[Path], int]:
        """
        Return (new_paths, skipped_count).

        A file is considered 'existing' if its SHA-256 hash is already in the store.
        This handles renames correctly: a renamed file with the same content is
        already indexed and won't be re-embedded.
        """
        new_paths: list[Path] = []
        skipped = 0

        for path in image_paths:
            try:
                file_hash = self._hash_file(path)
            except (OSError, PermissionError):
                skipped += 1
                continue

            if file_hash in existing_hashes:
                skipped += 1
            else:
                new_paths.append(path)

        return new_paths, skipped

    def _index_batch(self, batch_paths: list[Path]) -> dict[str, int]:
        """
        Embed and store a batch of images.

        Returns dict with 'indexed' and 'errors' counts.
        """
        # Compute hashes (skip files we can't read)
        records: list[dict] = []
        paths_to_embed: list[Path] = []
        errors = 0

        for path in batch_paths:
            try:
                file_hash = self._hash_file(path)
                records.append({
                    "file_hash": file_hash,
                    "file_path": str(path),
                    "tags": [],
                    "date_taken": _extract_date(path),
                })
                paths_to_embed.append(path)
            except (OSError, PermissionError) as exc:
                logger.warning(f"Cannot hash {path.name}: {exc}")
                errors += 1

        if not paths_to_embed:
            return {"indexed": 0, "errors": errors}

        # Embed
        try:
            embeddings, successful_paths = self.embedder.encode_images(paths_to_embed)
        except Exception as exc:
            logger.error(f"Embedding batch failed: {exc}")
            return {"indexed": 0, "errors": errors + len(paths_to_embed)}

        if len(successful_paths) == 0:
            return {"indexed": 0, "errors": errors + len(paths_to_embed)}

        # Align records with successful paths
        path_to_record: dict[str, dict] = {r["file_path"]: r for r in records}
        successful_records = [path_to_record[str(p)] for p in successful_paths]

        errors += len(paths_to_embed) - len(successful_paths)

        # Persist to store
        self.store.add_batch(successful_records, embeddings)

        return {"indexed": len(successful_paths), "errors": errors}

    @staticmethod
    def _hash_file(path: Path) -> str:
        """Compute SHA-256 hash of a file's contents (64-char hex string)."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65_536):
                h.update(chunk)
        return h.hexdigest()
