---
title: Archivist AI
emoji: 🗂️
colorFrom: orange
colorTo: gray
sdk: gradio
sdk_version: "4.44.0"
app_file: app_hf.py
pinned: false
license: mit
short_description: Local-first AI image search — natural language, reverse image, duplicates
tags:
  - image-search
  - clip
  - siglip
  - faiss
  - privacy
  - computer-vision
  - gradio
---

# 🗂️ Archivist-AI

**Search your image library with plain English. No cloud. No API keys. 100% private.**

This Space is a **live demo** of [Archivist-AI](https://github.com/yourusername/archivist-ai) — a fully local, open-source AI image search tool.

> ⚠️ **Note:** This demo indexes a small sample image set. To search your own photos, install the package locally (see below) — your images never leave your machine.

---

## What it does

- **Natural language search** — `"sunset over water"`, `"birthday party with kids"`, `"people in business suits"`
- **Reverse image search** — upload any image to find visually similar ones
- **Duplicate detection** — find near-identical images using perceptual similarity
- **Date filtering** — search within a specific time range using EXIF dates

Powered by [SigLIP](https://huggingface.co/google/siglip-base-patch16-224) (Google's vision-language model) and [FAISS](https://github.com/facebookresearch/faiss) vector search.

---

## Run it on your own machine

```bash
pip install archivist-ai
archivist index ~/Pictures
archivist ui
```

Your photos stay on your computer. Nothing is uploaded.

**GitHub:** [github.com/yourusername/archivist-ai](https://github.com/yourusername/archivist-ai)
