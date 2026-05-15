# Changelog

All notable changes to Archivist-AI are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.1.0] — 2025-05-15

### Added
- Natural language image search via SigLIP (Google's vision-language model)
- Reverse image search (image-to-image)
- Incremental indexing with SHA-256 deduplication
- FAISS vector index with SQLite metadata store
- EXIF date extraction with mtime fallback
- Date range filtering on all search operations
- Near-duplicate detection using cosine similarity + Union-Find clustering
- Zero-shot auto-tagging with a built-in vocabulary
- Smart organiser: copy, move, and rename search results
- Real-time folder watcher for auto-indexing
- ONNX model export for 3–5× faster CPU inference
- Full CLI (`archivist index`, `search`, `similar`, `dupes`, `tag`, `copy`, `watch`, `clean`, `stats`, `export-onnx`, `ui`)
- Gradio web UI with Text Search, Reverse Image Search, Duplicate Finder, Index Folder, and Stats tabs
- Cross-platform CI (Windows, macOS, Linux × Python 3.9–3.12)
- MIT License
