"""
Search engine for Archivist-AI.

Provides three search modes:
1. Text-to-image   : "people in suits"  →  ranked image results
2. Image-to-image  : query_image.jpg    →  visually similar images
3. Multi-query     : ["sunset", "beach"] →  union / intersection of results

Score normalisation
-------------------
Raw cosine similarity scores vary significantly between embedding models:
- CLIP ViT-B/32  : matching pairs typically score 0.25–0.50
- SigLIP         : matching pairs typically score 0.10–0.28

To make the Min Similarity threshold meaningful regardless of the model,
scores are min-max normalised per query so the best match always scores
1.0 and the rest are relative to it. A threshold of 0.5 therefore means
"at least half as relevant as the best result" — intuitive for any model.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from .embedder import BaseEmbedder
from .store import ArchivistStore, SearchResult

logger = logging.getLogger(__name__)


class ArchivistSearcher:
    """
    High-level search interface backed by FAISS + CLIP.

    Example
    -------
    >>> searcher = ArchivistSearcher(store, embedder)
    >>> results = searcher.search_by_text("golden retriever playing in snow", k=10)
    >>> for r in results:
    ...     print(r.rank, r.similarity, r.file_path)
    """

    def __init__(self, store: ArchivistStore, embedder: BaseEmbedder) -> None:
        self.store = store
        self.embedder = embedder

    # ── Score normalisation ───────────────────────────────────────────────────

    @staticmethod
    def _normalise_scores(results: list[SearchResult]) -> list[SearchResult]:
        """
        Min-max normalise similarity scores so the best result = 1.0.

        This makes the threshold slider model-agnostic: 0.5 always means
        "at least half as relevant as the top result", whether the backend
        is CLIP (higher raw scores) or SigLIP (lower raw scores).
        """
        if not results:
            return results
        scores = [r.similarity for r in results]
        hi, lo = max(scores), min(scores)
        span = hi - lo
        for r in results:
            r.similarity = round((r.similarity - lo) / span, 4) if span > 1e-8 else 1.0
        return results

    # ── Text search ───────────────────────────────────────────────────────────

    def search_by_text(
        self,
        query: str,
        k: int = 20,
        threshold: float = 0.0,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[SearchResult]:
        """
        Natural language search.

        Parameters
        ----------
        query     : Free-form description, e.g. "birthday cake with candles".
        k         : Maximum results to return.
        threshold : Minimum cosine similarity (0–1). Filters out weak matches.
                    Try 0.20 for stricter results.
        date_from : Optional ISO date "YYYY-MM-DD" — earliest image date to include.
        date_to   : Optional ISO date "YYYY-MM-DD" — latest image date to include.

        Returns
        -------
        Ranked list of SearchResult, best match first.
        """
        query = query.strip()
        if not query:
            return []

        query_emb = self.embedder.encode_text([query])[0]
        results = self.store.search(
            query_emb, k=k, date_from=date_from or None, date_to=date_to or None
        )
        results = self._normalise_scores(results)

        if threshold > 0.0:
            results = [r for r in results if r.similarity >= threshold]

        logger.debug(f"Text search '{query}': {len(results)} results")
        return results

    # ── Image-to-image search ─────────────────────────────────────────────────

    def search_by_image(
        self,
        image_path: Path | str,
        k: int = 20,
        threshold: float = 0.0,
        exclude_self: bool = True,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[SearchResult]:
        """
        Reverse image search — find images visually similar to a given image.

        Parameters
        ----------
        image_path   : Path to the query image (does not need to be indexed).
        k            : Maximum results.
        threshold    : Minimum cosine similarity.
        exclude_self : Skip the query image itself if it is in the index.
        date_from    : Optional ISO date "YYYY-MM-DD" — earliest image date to include.
        date_to      : Optional ISO date "YYYY-MM-DD" — latest image date to include.

        Returns
        -------
        Ranked list of SearchResult, most similar first.
        """
        image_path = Path(image_path).expanduser().resolve()

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Cannot open query image {image_path}: {exc}") from exc

        query_emb = self.embedder.encode_pil_images([img])[0]

        # Request k+1 so we have room to exclude self without truncating results
        results = self.store.search(
            query_emb,
            k=k + (1 if exclude_self else 0),
            date_from=date_from or None,
            date_to=date_to or None,
        )

        if exclude_self:
            results = [r for r in results if r.file_path.resolve() != image_path]

        results = self._normalise_scores(results)

        if threshold > 0.0:
            results = [r for r in results if r.similarity >= threshold]

        # Re-rank after filtering
        for i, r in enumerate(results[:k]):
            r.rank = i + 1

        return results[:k]

    # ── Multi-query search ────────────────────────────────────────────────────

    def multi_query_search(
        self,
        queries: list[str],
        k: int = 20,
        threshold: float = 0.0,
        merge: str = "union",
    ) -> list[SearchResult]:
        """
        Search with multiple text queries and merge results.

        Parameters
        ----------
        queries   : List of text queries.
        k         : Max results per query.
        threshold : Minimum cosine similarity.
        merge     : 'union'  — all results from any query (deduplicated, re-ranked).
                    'intersect' — only results appearing in ALL queries.

        Returns
        -------
        Deduplicated, re-ranked list of SearchResult.
        """
        if not queries:
            return []

        # Collect per-query result sets (faiss_id → best SearchResult)
        all_sets: list[dict[int, SearchResult]] = []

        for query in queries:
            query_map: dict[int, SearchResult] = {}
            for r in self.search_by_text(query, k=k, threshold=threshold):
                query_map[r.faiss_id] = r
            all_sets.append(query_map)

        if merge == "intersect":
            # Keep only IDs present in every query
            common_ids = set(all_sets[0].keys())
            for s in all_sets[1:]:
                common_ids &= s.keys()

            merged: dict[int, SearchResult] = {}
            for fid in common_ids:
                # Average similarity across queries
                sims = [s[fid].similarity for s in all_sets if fid in s]
                best = all_sets[0][fid]
                best.similarity = float(np.mean(sims))
                merged[fid] = best
        else:  # union
            merged = {}
            for query_map in all_sets:
                for fid, result in query_map.items():
                    if fid not in merged:
                        merged[fid] = result
                    else:
                        # Take the higher similarity (or average — both reasonable)
                        merged[fid].similarity = max(
                            merged[fid].similarity, result.similarity
                        )

        results = sorted(merged.values(), key=lambda r: r.similarity, reverse=True)

        for i, r in enumerate(results[:k]):
            r.rank = i + 1

        return results[:k]

    # ── Convenience ───────────────────────────────────────────────────────────

    def find_similar(
        self,
        image_path: Path | str,
        k: int = 10,
    ) -> list[SearchResult]:
        """Alias for search_by_image with exclude_self=True. Cleaner API."""
        return self.search_by_image(image_path, k=k, exclude_self=True)
