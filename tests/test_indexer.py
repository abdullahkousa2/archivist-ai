"""Tests for ArchivistIndexer — incremental indexing logic."""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from archivist.core.indexer import ArchivistIndexer, IndexResult, SUPPORTED_EXTENSIONS
from archivist.core.store import ArchivistStore


def _fake_embedder(dim: int = 512):
    """Mock embedder that returns random unit vectors."""
    embedder = MagicMock()
    embedder.EMBEDDING_DIM = dim

    def encode_images(paths):
        vecs = np.random.randn(len(paths), dim).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs, list(paths)

    embedder.encode_images.side_effect = encode_images
    return embedder


@pytest.fixture
def image_dir(tmp_path):
    """Create a fake image directory with some .jpg files."""
    imgs = []
    for i in range(5):
        p = tmp_path / f"photo_{i:02d}.jpg"
        p.write_bytes(b"\xff\xd8\xff" + bytes([i] * 100))  # Minimal fake JPEG bytes
        imgs.append(p)
    return tmp_path, imgs


@pytest.fixture
def store(tmp_path):
    s = ArchivistStore(tmp_path / "index")
    s.open()
    yield s
    s.close()


class TestCollectImages:
    def test_collects_supported_extensions(self, tmp_path):
        embedder = _fake_embedder()
        store = MagicMock()
        indexer = ArchivistIndexer(store, embedder)

        # Create files with various extensions
        (tmp_path / "a.jpg").touch()
        (tmp_path / "b.PNG").touch()
        (tmp_path / "c.webp").touch()
        (tmp_path / "d.txt").touch()  # Should be ignored
        (tmp_path / "e.mp4").touch()  # Should be ignored

        paths = indexer._collect_images(tmp_path, recursive=False)
        suffixes = {p.suffix.lower() for p in paths}
        assert ".jpg" in suffixes
        assert ".png" in suffixes
        assert ".webp" in suffixes
        assert ".txt" not in suffixes
        assert ".mp4" not in suffixes

    def test_recursive_vs_flat(self, tmp_path):
        embedder = _fake_embedder()
        store = MagicMock()
        indexer = ArchivistIndexer(store, embedder)

        (tmp_path / "top.jpg").touch()
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "nested.jpg").touch()

        flat = indexer._collect_images(tmp_path, recursive=False)
        deep = indexer._collect_images(tmp_path, recursive=True)

        assert len(flat) == 1
        assert len(deep) == 2


class TestHashFile:
    def test_hash_deterministic(self, tmp_path):
        p = tmp_path / "test.jpg"
        p.write_bytes(b"hello world")
        h1 = ArchivistIndexer._hash_file(p)
        h2 = ArchivistIndexer._hash_file(p)
        assert h1 == h2

    def test_hash_changes_with_content(self, tmp_path):
        p = tmp_path / "test.jpg"
        p.write_bytes(b"content_a")
        h1 = ArchivistIndexer._hash_file(p)
        p.write_bytes(b"content_b")
        h2 = ArchivistIndexer._hash_file(p)
        assert h1 != h2

    def test_hash_is_sha256(self, tmp_path):
        content = b"archivist test"
        p = tmp_path / "test.jpg"
        p.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert ArchivistIndexer._hash_file(p) == expected


class TestIncrementalIndexing:
    def test_first_run_indexes_all(self, image_dir, store):
        directory, imgs = image_dir
        embedder = _fake_embedder()
        indexer = ArchivistIndexer(store, embedder, batch_size=2)

        result = indexer.index_directory(directory, show_progress=False)

        assert result.newly_indexed == 5
        assert result.skipped_existing == 0
        assert result.total_found == 5
        assert store.count() == 5

    def test_second_run_skips_all(self, image_dir, store):
        directory, imgs = image_dir
        embedder = _fake_embedder()
        indexer = ArchivistIndexer(store, embedder)

        indexer.index_directory(directory, show_progress=False)
        result2 = indexer.index_directory(directory, show_progress=False)

        assert result2.newly_indexed == 0
        assert result2.skipped_existing == 5

    def test_new_file_detected(self, image_dir, store):
        directory, imgs = image_dir
        embedder = _fake_embedder()
        indexer = ArchivistIndexer(store, embedder)

        indexer.index_directory(directory, show_progress=False)

        # Add a new image
        new_img = directory / "new_photo.jpg"
        new_img.write_bytes(b"\xff\xd8\xff" + b"\xAA" * 100)

        result2 = indexer.index_directory(directory, show_progress=False)
        assert result2.newly_indexed == 1
        assert result2.skipped_existing == 5

    def test_index_multiple_directories(self, tmp_path, store):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        for i in range(3):
            (dir_a / f"a{i}.jpg").write_bytes(bytes([i] * 50))
            (dir_b / f"b{i}.jpg").write_bytes(bytes([i + 100] * 50))

        embedder = _fake_embedder()
        indexer = ArchivistIndexer(store, embedder)
        result = indexer.index_multiple_directories([dir_a, dir_b], show_progress=False)

        assert result.total_found == 6
        assert result.newly_indexed == 6

    def test_invalid_directory_raises(self, store):
        embedder = _fake_embedder()
        indexer = ArchivistIndexer(store, embedder)
        with pytest.raises(ValueError):
            indexer.index_directory(Path("/nonexistent/path"), show_progress=False)
