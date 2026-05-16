"""
Archivist-AI — HuggingFace Space demo (gradio 5).
"""

from __future__ import annotations
import urllib.request
from pathlib import Path

# ── Sample images ─────────────────────────────────────────────────────────────
SAMPLE_IMAGES = {
    "dog.jpg":      "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=400",
    "cat.jpg":      "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400",
    "mountain.jpg": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=400",
    "beach.jpg":    "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=400",
    "forest.jpg":   "https://images.unsplash.com/photo-1448375240586-882707db888b?w=400",
    "city.jpg":     "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=400",
    "snow.jpg":     "https://images.unsplash.com/photo-1418985991508-e47386d96a71?w=400",
    "food.jpg":     "https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=400",
    "people.jpg":   "https://images.unsplash.com/photo-1529156069898-49953e39b3ac?w=400",
    "sunset.jpg":   "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400",
    "bicycle.jpg":  "https://images.unsplash.com/photo-1485965120184-e220f721d03e?w=400",
}

SAMPLES_DIR = Path("/tmp/archivist_samples")
INDEX_DIR   = Path("/tmp/archivist_index")


def download_samples():
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in SAMPLE_IMAGES.items():
        dest = SAMPLES_DIR / filename
        if dest.exists():
            continue
        print(f"Downloading {filename}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                dest.write_bytes(r.read())
        except Exception as e:
            print(f"  Could not download {filename}: {e}")


def setup():
    from archivist.config import ArchivistConfig
    from archivist.core import ArchivistIndexer, ArchivistStore, ArchivistSearcher
    from archivist.core.embedder import create_embedder

    cfg = ArchivistConfig(
        index_dir=INDEX_DIR,
        model_id="openai/clip-vit-base-patch32",
        embedding_dim=512,
        device="cpu",
        quantize=True,
        batch_size=4,
    )
    embedder = create_embedder(
        model_id=cfg.model_id,
        device=cfg.device,
        batch_size=cfg.batch_size,
        quantize=cfg.quantize,
    )
    store = ArchivistStore(cfg.index_dir, embedding_dim=cfg.embedding_dim)
    store.open()
    if store.count() < len(SAMPLE_IMAGES):
        indexer = ArchivistIndexer(store, embedder, batch_size=cfg.batch_size)
        indexer.index_directory(SAMPLES_DIR, show_progress=True)
        print(f"Indexed {store.count()} images.")
    return ArchivistSearcher(store, embedder)


print("Setting up Archivist-AI demo...")
download_samples()
searcher = setup()
print("Ready — launching UI.")

# ── Gradio UI (gradio 5) ───────────────────────────────────────────────────────
import gradio as gr


def search(query: str, k: int, threshold: float):
    if not query.strip():
        return [], "Enter a search query."
    results = searcher.search_by_text(query, k=int(k), threshold=float(threshold))
    if not results:
        return [], f"No results for: '{query}'"
    rows = [(str(r.file_path), f"#{r.rank} · {r.similarity:.3f} · {r.filename}")
            for r in results if r.file_path.exists()]
    return rows, f"✓ {len(rows)} results for: '{query}'"


def reverse_search(img, k: int, threshold: float):
    if img is None:
        return [], "Upload a query image."
    import tempfile
    from PIL import Image as PILImage
    import numpy as np
    # gradio 5 passes numpy arrays for type="numpy"
    arr = img if isinstance(img, np.ndarray) else np.array(img)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        PILImage.fromarray(arr).save(f.name)
        tmp = Path(f.name)
    try:
        results = searcher.search_by_image(tmp, k=int(k), threshold=float(threshold))
    finally:
        tmp.unlink(missing_ok=True)
    if not results:
        return [], "No similar images found."
    rows = [(str(r.file_path), f"#{r.rank} · {r.similarity:.3f} · {r.filename}")
            for r in results if r.file_path.exists()]
    return rows, f"✓ {len(rows)} similar images found"


with gr.Blocks(title="Archivist-AI") as demo:
    gr.Markdown("""
# 🗂️ Archivist-AI Demo
**Local-first AI image search — powered by CLIP. No cloud. No API keys.**
> This demo searches 11 sample images. [Install locally](https://github.com/abdullahkousa2/archivist-ai) to search your own photos.

Try: `dog`, `mountain`, `food`, `people`, `sunset`, `bicycle`
""")
    with gr.Tabs():
        with gr.Tab("🔍 Text Search"):
            with gr.Row():
                with gr.Column(scale=1):
                    t_query   = gr.Textbox(label="Search Query", placeholder="dog, mountain, sunset...")
                    t_k       = gr.Slider(1, 11, value=6, step=1, label="Max Results")
                    t_thresh  = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Min Similarity")
                    t_btn     = gr.Button("Search", variant="primary")
                    t_status  = gr.Markdown()
                with gr.Column(scale=3):
                    t_gallery = gr.Gallery(label="Results", columns=3, height=420, object_fit="cover")
            t_btn.click(search, [t_query, t_k, t_thresh], [t_gallery, t_status])
            t_query.submit(search, [t_query, t_k, t_thresh], [t_gallery, t_status])

        with gr.Tab("🖼️ Reverse Image Search"):
            with gr.Row():
                with gr.Column(scale=1):
                    i_img    = gr.Image(label="Upload Query Image", type="numpy", height=220)
                    i_k      = gr.Slider(1, 11, value=5, step=1, label="Max Results")
                    i_thresh = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Min Similarity")
                    i_btn    = gr.Button("Find Similar", variant="primary")
                    i_status = gr.Markdown()
                with gr.Column(scale=3):
                    i_gallery = gr.Gallery(label="Similar Images", columns=3, height=420, object_fit="cover")
            i_btn.click(reverse_search, [i_img, i_k, i_thresh], [i_gallery, i_status])

demo.launch()
