"""
Near-duplicate image detection for Archivist-AI.

Algorithm
---------
1. Retrieve all stored embeddings from FAISS (already L2-normalised).
2. Build a temporary FAISS index and run range_search with the duplicate
   threshold (default 0.97 cosine similarity).
3. Use Union-Find to group connected duplicate pairs into clusters.
4. Return DuplicateGroup objects sorted by similarity (worst duplication first).

Why cosine similarity, not pixel-level comparison?
- Captures perceptual similarity: cropped, resized, colour-adjusted, or
  JPEG-re-encoded versions of the same photo are correctly grouped.
- Operates on already-computed embeddings — zero additional model inference.

Threshold guide
---------------
  0.99+  : Visually identical (different format / compression only)
  0.97   : Very similar (minor crop, rotation, filter) ← default
  0.90   : Same scene / subject, different angle
  0.80   : Loosely similar (same category)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np

from archivist.core.store import ArchivistStore

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class DuplicateGroup:
    """A cluster of near-duplicate images."""

    group_id: int
    images: list[Path]
    max_similarity: float

    @property
    def size(self) -> int:
        return len(self.images)

    @property
    def potential_savings(self) -> str:
        """Human-readable estimate of space freed by keeping only one image."""
        try:
            sizes = [p.stat().st_size for p in self.images if p.exists()]
            if not sizes:
                return "unknown"
            saveable = sum(sizes) - min(sizes)
            if saveable >= 1_000_000_000:
                return f"{saveable / 1e9:.1f} GB"
            if saveable >= 1_000_000:
                return f"{saveable / 1e6:.1f} MB"
            return f"{saveable / 1e3:.0f} KB"
        except OSError:
            return "unknown"

    def __repr__(self) -> str:
        return (
            f"DuplicateGroup("
            f"id={self.group_id}, "
            f"size={self.size}, "
            f"similarity={self.max_similarity:.4f}, "
            f"savings={self.potential_savings})"
        )


# ── Finder ────────────────────────────────────────────────────────────────────


class DuplicateFinder:
    """
    Finds near-duplicate image clusters across the entire index.

    Example
    -------
    >>> finder = DuplicateFinder(store, threshold=0.97)
    >>> groups = finder.find_duplicates()
    >>> for g in groups:
    ...     print(g.size, "dupes — could save", g.potential_savings)
    """

    def __init__(self, store: ArchivistStore, threshold: float = 0.97) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        self.store = store
        self.threshold = threshold

    # ── Public API ────────────────────────────────────────────────────────────

    def find_duplicates(self) -> list[DuplicateGroup]:
        """
        Scan the entire index and return groups of near-duplicate images.

        Returns
        -------
        List of DuplicateGroup (each ≥ 2 images), sorted by max_similarity
        descending (most exact duplicates first).
        """
        all_vecs, all_ids = self.store.get_all_vectors()
        n = len(all_ids)

        if n < 2:
            logger.info("Not enough images in index to detect duplicates.")
            return []

        logger.info(
            f"Scanning {n} images for near-duplicates "
            f"(threshold={self.threshold:.2f})..."
        )

        # Build a temporary FAISS index for range search
        dim = all_vecs.shape[1]
        tmp_index = faiss.IndexFlatIP(dim)
        tmp_index.add(all_vecs)

        # Range search: find all pairs with similarity >= threshold
        lims, D, I = tmp_index.range_search(all_vecs, self.threshold)

        # Union-Find to group connected components
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # Path compression
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            start, end = int(lims[i]), int(lims[i + 1])
            for j in I[start:end]:
                if i != j:
                    union(i, j)

        # Collect groups (only those with ≥ 2 members)
        groups_map: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            groups_map.setdefault(root, []).append(i)

        # Map FAISS vector indices → file paths via SQLite
        db_rows = self.store._db.execute(
            "SELECT faiss_id, file_path FROM images ORDER BY faiss_id"
        ).fetchall()
        id_to_path: dict[int, Path] = {
            row["faiss_id"]: Path(row["file_path"]) for row in db_rows
        }

        result: list[DuplicateGroup] = []
        for group_id, (root, members) in enumerate(groups_map.items()):
            if len(members) < 2:
                continue

            paths = [id_to_path[i] for i in members if i in id_to_path]
            if len(paths) < 2:
                continue

            # Compute max pairwise similarity within group
            group_vecs = all_vecs[members]
            sim_matrix = group_vecs @ group_vecs.T
            np.fill_diagonal(sim_matrix, 0.0)
            max_sim = float(sim_matrix.max())

            result.append(DuplicateGroup(
                group_id=group_id,
                images=paths,
                max_similarity=max_sim,
            ))

        result.sort(key=lambda g: g.max_similarity, reverse=True)
        logger.info(f"Found {len(result)} duplicate group(s)")
        return result
