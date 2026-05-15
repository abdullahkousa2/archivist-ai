"""Tests for ArchivistStore (FAISS + SQLite layer)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from archivist.core.store import ArchivistStore, SearchResult


@pytest.fixture
def tmp_store(tmp_path):
    """Open a fresh store in a temporary directory."""
    store = ArchivistStore(tmp_path / "index", embedding_dim=512)
    store.open()
    yield store
    store.close()


def _random_unit_vec(dim: int = 512) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _dummy_path(name: str = "img.jpg") -> Path:
    return Path(f"/fake/path/{name}")


class TestStoreLifecycle:
    def test_open_creates_files(self, tmp_path):
        store = ArchivistStore(tmp_path / "idx", embedding_dim=512)
        store.open()
        assert store.db_path.exists()
        store.close()

    def test_context_manager(self, tmp_path):
        with ArchivistStore(tmp_path / "idx") as store:
            assert store.count() == 0

    def test_count_empty(self, tmp_store):
        assert tmp_store.count() == 0

    def test_stats_keys(self, tmp_store):
        s = tmp_store.stats()
        assert "total_indexed" in s
        assert "faiss_vectors" in s
        assert "embedding_dim" in s


class TestAddAndSearch:
    def test_add_single(self, tmp_store, tmp_path):
        vec = _random_unit_vec()
        fid = tmp_store.add("abc123", tmp_path / "a.jpg", vec)
        assert fid == 0
        assert tmp_store.count() == 1

    def test_add_duplicate_hash(self, tmp_store, tmp_path):
        vec = _random_unit_vec()
        fid1 = tmp_store.add("same_hash", tmp_path / "a.jpg", vec)
        fid2 = tmp_store.add("same_hash", tmp_path / "b.jpg", vec)
        assert fid1 == fid2
        assert tmp_store.count() == 1

    def test_add_batch(self, tmp_store, tmp_path):
        n = 10
        vecs = np.random.randn(n, 512).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        records = [{"file_hash": f"hash{i}", "file_path": str(tmp_path / f"img{i}.jpg"), "tags": []} for i in range(n)]
        ids = tmp_store.add_batch(records, vecs)
        assert len(ids) == n
        assert tmp_store.count() == n

    def test_search_returns_results(self, tmp_store, tmp_path):
        # Add 5 images
        for i in range(5):
            vec = _random_unit_vec()
            tmp_store.add(f"hash{i}", tmp_path / f"img{i}.jpg", vec)

        query = _random_unit_vec()
        results = tmp_store.search(query, k=3)
        assert len(results) <= 3
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_ranking(self, tmp_store, tmp_path):
        """The first result should have similarity >= subsequent results."""
        for i in range(20):
            tmp_store.add(f"hash{i}", tmp_path / f"img{i}.jpg", _random_unit_vec())

        query = _random_unit_vec()
        results = tmp_store.search(query, k=10)
        sims = [r.similarity for r in results]
        assert sims == sorted(sims, reverse=True), "Results should be in descending similarity order"

    def test_search_empty_store(self, tmp_store):
        results = tmp_store.search(_random_unit_vec(), k=5)
        assert results == []

    def test_result_rank_numbering(self, tmp_store, tmp_path):
        for i in range(5):
            tmp_store.add(f"hash{i}", tmp_path / f"img{i}.jpg", _random_unit_vec())
        results = tmp_store.search(_random_unit_vec(), k=5)
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(results) + 1))


class TestHashes:
    def test_get_all_hashes_empty(self, tmp_store):
        assert tmp_store.get_all_hashes() == set()

    def test_get_all_hashes_after_add(self, tmp_store, tmp_path):
        hashes = {"h1", "h2", "h3"}
        for h in hashes:
            tmp_store.add(h, tmp_path / f"{h}.jpg", _random_unit_vec())
        assert tmp_store.get_all_hashes() == hashes


class TestRemove:
    def test_remove_by_hash(self, tmp_store, tmp_path):
        vec = _random_unit_vec()
        tmp_store.add("todelete", tmp_path / "x.jpg", vec)
        assert tmp_store.count() == 1
        tmp_store.remove_by_hash("todelete")
        assert tmp_store.count() == 0

    def test_remove_missing_files(self, tmp_store, tmp_path):
        # Add a real file entry
        real_file = tmp_path / "real.jpg"
        real_file.touch()
        tmp_store.add("real_hash", real_file, _random_unit_vec())

        # Add a ghost file entry (doesn't exist on disk)
        tmp_store.add("ghost_hash", Path("/nonexistent/ghost.jpg"), _random_unit_vec())

        assert tmp_store.count() == 2
        removed = tmp_store.remove_missing_files()
        assert removed == 1
        assert tmp_store.count() == 1


class TestTags:
    def test_update_tags(self, tmp_store, tmp_path):
        fid = tmp_store.add("tagged", tmp_path / "a.jpg", _random_unit_vec())
        tmp_store.update_tags(fid, [("sunset", 0.9), ("beach", 0.7)])

        record = tmp_store.get_record(fid)
        assert "sunset" in record.tags
        assert "beach" in record.tags

    def test_replace_tags(self, tmp_store, tmp_path):
        fid = tmp_store.add("tagged2", tmp_path / "b.jpg", _random_unit_vec())
        tmp_store.update_tags(fid, [("old_tag", 0.8)])
        tmp_store.update_tags(fid, [("new_tag", 0.9)], replace=True)

        record = tmp_store.get_record(fid)
        assert "new_tag" in record.tags
        assert "old_tag" not in record.tags


class TestPersistence:
    def test_reload_after_save(self, tmp_path):
        """FAISS index should be loadable from disk in a new Store instance."""
        vec = _random_unit_vec()
        index_dir = tmp_path / "idx"

        with ArchivistStore(index_dir) as store:
            store.add("persisted", tmp_path / "img.jpg", vec)
            assert store.count() == 1

        # Reload from disk
        with ArchivistStore(index_dir) as store2:
            assert store2.count() == 1
            results = store2.search(vec, k=1)
            assert len(results) == 1
            assert results[0].similarity > 0.99  # Should find the same vector
