<div align="center">

# 🗂️ Archivist-AI

### Local-first AI image search & management — no cloud, no API keys, 100% private.

[![CI](https://github.com/yourusername/archivist-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/archivist-ai/actions)
[![PyPI](https://img.shields.io/pypi/v/archivist-ai?color=orange)](https://pypi.org/project/archivist-ai/)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue)](https://pypi.org/project/archivist-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)](https://github.com/yourusername/archivist-ai)
[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-yellow)](https://huggingface.co/spaces/yourusername/archivist-ai)

**Search your entire photo library with plain English. Runs fully offline on any CPU.**

[Quick Start](#-quick-start) · [Features](#-features) · [Web UI](#️-web-ui) · [CLI](#-cli-reference) · [How It Works](#-how-it-works) · [Roadmap](#️-roadmap)

</div>

---

## Why Archivist-AI?

Most photo search tools send your images to the cloud. Google Photos, Apple Photos, and Amazon Photos all require accounts, upload your data to remote servers, and lock you into their ecosystem.

**Archivist-AI runs entirely on your machine.** Your photos never leave your computer.

| | Archivist-AI | Google Photos | Apple Photos |
|---|:---:|:---:|:---:|
| Works offline | ✅ | ❌ | ❌ |
| No account needed | ✅ | ❌ | ❌ |
| Images stay on your machine | ✅ | ❌ | ❌ |
| Natural language search | ✅ | ✅ | ✅ |
| Reverse image search | ✅ | ✅ | ❌ |
| Duplicate detection | ✅ | ✅ | ❌ |
| Open source | ✅ | ❌ | ❌ |
| Works on any folder | ✅ | ❌ | ❌ |

---

## ✨ Features

- **🔍 Natural language search** — Type `"birthday cake with candles"` or `"sunset over mountains"` and find the right photo instantly. Powered by [SigLIP](https://huggingface.co/google/siglip-base-patch16-224) (Google's state-of-the-art vision-language model).
- **🖼️ Reverse image search** — Drag in any image to find visually similar photos in your library.
- **🔎 Duplicate detection** — Finds near-duplicate images using perceptual similarity — catches re-encoded, cropped, or slightly edited copies.
- **🏷️ Zero-shot auto-tagging** — Automatically tag images using natural categories (`portrait`, `sunset`, `dog`, `indoor`) with no training required.
- **📁 Smart organiser** — Copy or move search results to a new folder, or rename them by query.
- **👁️ Folder watcher** — Monitor directories and auto-index new images in real time.
- **📅 Date filtering** — Filter searches by EXIF date or file modification date.
- **⚡ ONNX acceleration** — Export the model to ONNX for 3–5× faster CPU inference.
- **🖥️ Gradio web UI** — A clean local browser interface for all features.
- **⌨️ Full CLI** — Scriptable, composable, pipe-friendly.

---

## 🚀 Quick Start

### 1. Install

```bash
pip install archivist-ai
```

> **Requirements:** Python 3.9+. No GPU needed.

### 2. Index your photos

```bash
archivist index ~/Pictures
```

The first run downloads the SigLIP model (~375 MB, once). Subsequent runs only process new images.

### 3. Search

```bash
archivist search "people laughing at a dinner table"
```

### 4. Launch the web UI

```bash
archivist ui
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

---

## 🖥️ Web UI

Launch with `archivist ui` and get a full-featured browser interface:

| Tab | What it does |
|---|---|
| 🔍 **Text Search** | Natural language search with similarity threshold and date filters |
| 🖼️ **Reverse Image Search** | Upload any image to find visually similar ones |
| 🔎 **Find Duplicates** | Scan for near-duplicates and delete extras with one click |
| 📁 **Index Folder** | Add a new folder to the index from the browser |
| 📊 **Stats** | Index size, date range, storage breakdown |

---

## ⌨️ CLI Reference

```
archivist index <dirs...>        Index image directories (incremental)
archivist search <query>         Natural language search
archivist similar <image>        Reverse image search
archivist dupes                  Find near-duplicate images
archivist tag                    Auto-tag all untagged images
archivist copy <query> <dest>    Copy search results to a folder
archivist watch <dirs...>        Watch folders and auto-index new arrivals
archivist clean                  Remove stale entries for deleted files
archivist stats                  Show index statistics
archivist export-onnx            Export model to ONNX (3–5× faster)
archivist ui                     Launch the Gradio web UI
```

**Examples:**

```bash
# Search with stricter threshold and more results
archivist search "cats playing" --top-k 50 --threshold 0.3

# Index multiple folders, non-recursive
archivist index ~/Photos ~/Downloads --no-recursive

# Find only near-identical duplicates
archivist dupes --threshold 0.99

# Preview what would be copied without doing it
archivist copy "wedding photos" ~/Desktop/Wedding --dry-run

# Watch a folder and auto-index as new photos arrive
archivist watch ~/Downloads
```

---

## ⚡ Speed: ONNX Mode

For significantly faster indexing and search on CPU:

```bash
# Export the model once (takes ~1 minute)
archivist export-onnx

# All subsequent commands use ONNX automatically
archivist search "golden retriever"
```

ONNX mode enables int8 quantization and skips PyTorch entirely at inference time.

| Mode | ~Time per image |
|---|---|
| SigLIP (PyTorch, default) | ~0.30 s |
| SigLIP + quantization | ~0.15 s |
| ONNX (after export) | ~0.06–0.10 s |

---

## 🔧 How It Works

```
Your Photos
    │
    ▼
┌─────────────────────────────────────────┐
│  Indexer                                │
│  • SHA-256 dedup (skip unchanged files) │
│  • EXIF date extraction                 │
│  • SigLIP / ONNX embedding             │
└────────────────┬────────────────────────┘
                 │  768-dim float32 vector
        ┌────────▼────────┐
        │  FAISS Index    │  ← vector similarity search
        │  (IndexFlatIP)  │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │  SQLite DB      │  ← file path, hash, tags, date
        └────────┬────────┘
                 │
    ┌────────────▼──────────────────┐
    │  Query                        │
    │  "people in suits" ────────── │──▶ text embedding → FAISS → ranked results
    │  query_image.jpg ──────────── │──▶ image embedding → FAISS → ranked results
    └───────────────────────────────┘
```

**Why SigLIP over CLIP?**
SigLIP uses a sigmoid loss instead of softmax, making it significantly better at zero-shot retrieval — especially for complex or multi-concept queries. It's the model powering Google Lens.

**Why FAISS?**
Facebook's FAISS performs exact inner-product search in milliseconds even across 100,000+ images, with no server required.

**Incremental indexing:**
Files are identified by SHA-256 hash. Re-running `archivist index` on the same folder is near-instant — only new or changed files are embedded.

---

## 📦 Installation Options

**Stable (pip):**
```bash
pip install archivist-ai
```

**With ONNX acceleration:**
```bash
pip install "archivist-ai[onnx]"
```

**From source:**
```bash
git clone https://github.com/yourusername/archivist-ai
cd archivist-ai
pip install -e ".[dev]"
```

---

## ⚙️ Configuration

The config file lives at `~/.archivist/config.json` and is created automatically on first run.

```json
{
  "model_id": "google/siglip-base-patch16-224",
  "device": "cpu",
  "quantize": true,
  "use_onnx": false,
  "batch_size": 16,
  "top_k": 20,
  "duplicate_threshold": 0.97,
  "autotag_on_index": false
}
```

| Key | Default | Description |
|---|---|---|
| `model_id` | `google/siglip-base-patch16-224` | Vision-language model |
| `quantize` | `true` | Dynamic int8 quantization (faster, no quality loss) |
| `use_onnx` | `false` | Use ONNX runtime (run `export-onnx` first) |
| `batch_size` | `16` | Images per embedding batch |
| `duplicate_threshold` | `0.97` | Cosine similarity cutoff for duplicates |
| `autotag_on_index` | `false` | Auto-tag every image during indexing (slower) |

---

## 🗺️ Roadmap

- [ ] OCR search — find images containing specific text
- [ ] Face clustering — group photos by person (fully local)
- [ ] Smart albums — saved searches that auto-update
- [ ] Metadata editing — write tags back to EXIF
- [ ] Plugin API — bring your own embedder
- [ ] Desktop app (Electron/Tauri wrapper)

---

## 🤝 Contributing

Contributions are very welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

```bash
git clone https://github.com/yourusername/archivist-ai
cd archivist-ai
pip install -e ".[dev]"
pytest tests/
```

Please open an issue before submitting large PRs so we can discuss the approach first.

---

## 📄 License

MIT © 2025 — see [LICENSE](LICENSE) for details.

---

<div align="center">

**If Archivist-AI is useful to you, a ⭐ on GitHub goes a long way.**

Built for people who believe their photos belong to them.

</div>
