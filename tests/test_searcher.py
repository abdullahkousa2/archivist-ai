"""Tests for ArchivistSearcher — text and image search."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from archivist.core.searcher import ArchivistSearcher
from archivist.core.store import ArchivistStore, SearchResult


def _make_result(rank: int, faiss_id: int, similarity: float) -> SearchResult:
    return SearchResult(
        rank=rank,
        file_path=Path(f"/fake/img{faiss_id}.jpg"),
        filename=f"img{faiss_id}.jpg",
        similarity=similarity,
        tags=[],
        faiss_id=faiss_id,
    )


def _unit_vec(dim: int = 512) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def mock_store():
    store = MagicMock(spec=ArchivistStore)
    store.count.return_value = 10
    return store


@pytest.fixture
def mock_embedder():
    emb = MagicMock()
    emb.EMBEDDING_DIM = 512
    emb.encode_text.return_value = np.array([_unit_vec()])
    emb.encode_pil_images.return_value = np.array([_unit_vec()])
    return emb


@pytest.fixture
def searcher(mock_store, mock_embedder):
    return ArchivistSearcher(mock_store, mock_embedder)


class TestTextSearch:
    def test_empty_query_returns_empty(self, searcher):
        results = searcher.search_by_text("", k=10)
        assert results == []

    def test_whitespace_query_returns_empty(self, searcher):
        results = searcher.search_by_text("   ", k=10)
        assert results == []

    def test_calls_encode_text(self, searcher, mock_embedder, mock_store):
        mock_store.search.return_value = [_make_result(1, 0, 0.9)]
        searcher.search_by_text("sunset", k=5)
        mock_embedder.encode_text.assert_called_once_with(["sunset"])

    def test_threshold_filters_results(self, searcher, mock_store):
        mock_store.search.return_value = [
            _make_result(1, 0, 0.80),
            _make_result(2, 1, 0.60),
            _make_result(3, 2, 0.30),
        ]
        results = searcher.search_by_text("test", k=10, threshold=0.65)
        assert len(results) == 1
        assert results[0].similarity == 0.80

    def test_returns_all_when_threshold_zero(self, searcher, mock_store):
        mock_store.search.return_value = [
            _make_result(i + 1, i, 0.9 - i * 0.1) for i in range(5)
        ]
        results = searcher.search_by_text("test", k=10, threshold=0.0)
        assert len(results) == 5


class TestMultiQuerySearch:
    def test_union_deduplicates(self, searcher, mock_store):
        shared = _make_result(1, 0, 0.9)
        unique = _make_result(2, 1, 0.7)

        mock_store.search.side_effect = [
            [shared],
            [shared, unique],
        ]
        results = searcher.multi_query_search(["query1", "query2"], merge="union")
        faiss_ids = [r.faiss_id for r in results]
        assert len(faiss_ids) == len(set(faiss_ids)), "Union should deduplicate"

    def test_intersect_keeps_common_only(self, searcher, mock_store):
        common = _make_result(1, 0, 0.9)
        only_in_q1 = _make_result(2, 1, 0.8)
        only_in_q2 = _make_result(2, 2, 0.7)

        mock_store.search.side_effect = [
            [common, only_in_q1],
            [common, only_in_q2],
        ]
        results = searcher.multi_query_search(["q1", "q2"], merge="intersect")
        assert len(results) == 1
        assert results[0].faiss_id == 0

    def test_empty_queries_returns_empty(self, searcher):
        results = searcher.multi_query_search([], k=10)
        assert results == []
