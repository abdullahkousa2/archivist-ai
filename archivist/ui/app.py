"""
Gradio web UI for Archivist-AI.

Tabs
────
1. Text Search      — natural language → image grid with similarity scores
2. Reverse Search   — upload image → visually similar results
3. Duplicates       — scan for near-duplicate groups
4. Index            — add a new folder to the index
5. Explore (UMAP)   — interactive 2-D map of your entire library
6. Stats            — index statistics

Launch via CLI:  archivist ui
Or programmatically:
    from archivist.ui.app import launch
    launch(port=7860)
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from archivist.config import ArchivistConfig
from archivist.core import (
    ArchivistIndexer,
    ArchivistSearcher,
    ArchivistStore,
)
from archivist.core.embedder import create_embedder
from archivist.features.duplicates import DuplicateFinder

logger = logging.getLogger(__name__)


def _make_gallery(results) -> list[tuple[str, str]]:
    """Convert SearchResult list to Gradio gallery format (path, caption)."""
    rows = []
    for r in results:
        if not r.file_path.exists():
            continue
        date_str = f"  ·  {r.date_taken[:10]}" if r.date_taken else ""
        caption = f"#{r.rank}  ·  {r.similarity:.3f}  ·  {r.filename}{date_str}"
        rows.append((str(r.file_path), caption))
    return rows


def _parse_date(s: str) -> str | None:
    """Return stripped date string or None if empty."""
    s = s.strip()
    return s if s else None


def create_app(config: ArchivistConfig | None = None):
    """Build and return the Gradio Blocks app."""
    import gradio as gr

    if config is None:
        config = ArchivistConfig.load()

    # Shared components — backend auto-selected (SigLIP / CLIP / ONNX)
    if config.use_onnx:
        from archivist.core.onnx_embedder import ONNXEmbedder
        embedder = ONNXEmbedder(onnx_dir=config.onnx_dir, batch_size=config.batch_size)
    else:
        embedder = create_embedder(
            model_id=config.model_id,
            device=config.device,
            batch_size=config.batch_size,
            quantize=config.quantize,
        )
    store = ArchivistStore(config.index_dir, embedding_dim=config.embedding_dim)
    store.open()

    searcher = ArchivistSearcher(store, embedder)
    indexer = ArchivistIndexer(store, embedder, batch_size=config.batch_size)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def cb_text_search(query: str, k: int, threshold: float,
                       date_from: str, date_to: str):
        if not query.strip():
            return [], "⚠️  Enter a search query."
        if store.count() == 0:
            return [], "⚠️  Index is empty. Add a folder first."

        results = searcher.search_by_text(
            query,
            k=int(k),
            threshold=threshold,
            date_from=_parse_date(date_from),
            date_to=_parse_date(date_to),
        )
        if not results:
            filter_note = ""
            if _parse_date(date_from) or _parse_date(date_to):
                filter_note = " (try widening the date range)"
            return [], f"No results for: '{query}'{filter_note}"

        filter_note = ""
        if _parse_date(date_from) or _parse_date(date_to):
            filter_note = f"  ·  date filter active"
        return _make_gallery(results), f"✓  {len(results)} results for: '{query}'{filter_note}"

    def cb_image_search(img, k: int, threshold: float,
                        date_from: str, date_to: str):
        if img is None:
            return [], "⚠️  Upload a query image."
        if store.count() == 0:
            return [], "⚠️  Index is empty. Add a folder first."

        from PIL import Image as PILImage
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            PILImage.fromarray(img).save(f.name)
            tmp = Path(f.name)

        try:
            results = searcher.search_by_image(
                tmp,
                k=int(k),
                threshold=threshold,
                date_from=_parse_date(date_from),
                date_to=_parse_date(date_to),
            )
        finally:
            tmp.unlink(missing_ok=True)

        if not results:
            return [], "No similar images found."
        return _make_gallery(results), f"✓  {len(results)} similar images found"

    def cb_find_duplicates(threshold: float):
        import gradio as gr  # local import to use gr.update
        if store.count() < 2:
            return [], "⚠️  Need at least 2 images in the index.", gr.update(choices=[], value=[])

        finder = DuplicateFinder(store, threshold=threshold)
        groups = finder.find_duplicates()
        if not groups:
            return [], "✅  No near-duplicates found at this threshold.", gr.update(choices=[], value=[])

        gallery_items: list[tuple[str, str]] = []
        all_paths: list[str] = []
        extras: list[str] = []   # pre-checked — all but the first in each group

        for g in groups:
            for i, p in enumerate(g.images):
                path_str = str(p)
                caption = f"Group {g.group_id + 1}  ·  {p.name}  ·  sim {g.max_similarity:.4f}"
                if p.exists():
                    gallery_items.append((path_str, caption))
                all_paths.append(path_str)
                if i > 0:   # first image = "keep", rest = pre-checked for deletion
                    extras.append(path_str)

        total = sum(g.size for g in groups)
        status = (
            f"Found **{len(groups)} duplicate group(s)** across **{total} images**. "
            f"Extras are pre-checked — review then delete."
        )
        return gallery_items, status, gr.update(choices=all_paths, value=extras)

    def cb_delete_files(selected_paths: list[str], delete_from_disk: bool):
        """Remove selected files from the index and optionally from disk."""
        if not selected_paths:
            return "⚠️  No files selected."

        removed_index = 0
        removed_disk = 0
        errors: list[str] = []

        for path_str in selected_paths:
            p = Path(path_str)

            # Remove from search index
            faiss_id = store.get_faiss_id_by_path(p)
            if faiss_id is not None:
                record = store.get_record(faiss_id)
                if record:
                    store.remove_by_hash(record.file_hash)
                    removed_index += 1

            # Optionally delete from disk
            if delete_from_disk:
                try:
                    if p.exists():
                        p.unlink()
                        removed_disk += 1
                except OSError as exc:
                    errors.append(f"{p.name}: {exc}")

        if delete_from_disk:
            msg = (
                f"✅  **{removed_disk}** file(s) deleted from disk  "
                f"·  **{removed_index}** removed from index."
            )
        else:
            msg = f"✅  **{removed_index}** file(s) removed from index (files kept on disk)."

        if errors:
            msg += "\n\n⚠️  Errors:\n" + "\n".join(f"- {e}" for e in errors)

        return msg

    def cb_index_folder(folder_path: str):
        folder_path = folder_path.strip()
        if not folder_path:
            return "⚠️  Enter a directory path."
        p = Path(folder_path).expanduser()
        if not p.is_dir():
            return f"❌  Not a directory: {folder_path}"
        try:
            result = indexer.index_directory(p, show_progress=False)
            return (
                f"✅  **{result.newly_indexed}** new images indexed\n"
                f"⏭️  **{result.skipped_existing}** already in index\n"
                f"❌  **{result.skipped_errors}** errors\n"
                f"📦  **Total in index: {store.count()}**"
            )
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Indexing error: {exc}\n{tb}")
            return f"❌  Error: {exc}\n\n```\n{tb}\n```"

    def cb_stats():
        s = store.stats()
        min_d, max_d = store.get_date_range()
        date_line = (
            f"**Date range:** {min_d[:10]} → {max_d[:10]}\n"
            if min_d and max_d else ""
        )
        return (
            f"**Total indexed:** {s['total_indexed']}\n"
            f"**FAISS vectors:** {s['faiss_vectors']}\n"
            f"**Embedding dim:** {s['embedding_dim']}\n"
            f"{date_line}"
            f"**DB size:** {s['db_size_mb']} MB\n"
            f"**Index size:** {s['index_size_mb']} MB\n"
            f"**Location:** `{s['index_dir']}`"
        )

    # ── Layout ────────────────────────────────────────────────────────────────

    with gr.Blocks(title="Archivist-AI") as demo:
        gr.Markdown(
            """
            # 🗂️ Archivist-AI
            **Local-first AI image search — no cloud, no API keys, 100% private.**
            """,
            elem_classes="header-md",
        )

        with gr.Tabs():

            # ── Tab 1: Text search ─────────────────────────────────────────
            with gr.Tab("🔍 Text Search"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=240):
                        t_query = gr.Textbox(
                            label="Search Query",
                            placeholder="people in suits, sunset at the beach…",
                            lines=2,
                        )
                        t_k = gr.Slider(1, 100, value=20, step=1, label="Max Results")
                        t_thresh = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Min Similarity")
                        with gr.Row():
                            t_date_from = gr.Textbox(
                                label="Date From",
                                placeholder="YYYY-MM-DD",
                                scale=1,
                            )
                            t_date_to = gr.Textbox(
                                label="Date To",
                                placeholder="YYYY-MM-DD",
                                scale=1,
                            )
                        t_btn = gr.Button("Search", variant="primary")
                        t_status = gr.Markdown()
                    with gr.Column(scale=3):
                        t_gallery = gr.Gallery(
                            label="Results", columns=4, height=520, object_fit="cover"
                        )
                t_inputs = [t_query, t_k, t_thresh, t_date_from, t_date_to]
                t_btn.click(cb_text_search, t_inputs, [t_gallery, t_status])
                t_query.submit(cb_text_search, t_inputs, [t_gallery, t_status])

            # ── Tab 2: Reverse image search ────────────────────────────────
            with gr.Tab("🖼️ Reverse Image Search"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=280):
                        i_query = gr.Image(label="Upload Query Image", type="numpy", height=240)
                        i_k = gr.Slider(1, 100, value=10, step=1, label="Max Results")
                        i_thresh = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Min Similarity")
                        with gr.Row():
                            i_date_from = gr.Textbox(
                                label="Date From",
                                placeholder="YYYY-MM-DD",
                                scale=1,
                            )
                            i_date_to = gr.Textbox(
                                label="Date To",
                                placeholder="YYYY-MM-DD",
                                scale=1,
                            )
                        i_btn = gr.Button("Find Similar Images", variant="primary")
                        i_status = gr.Markdown()
                    with gr.Column(scale=3):
                        i_gallery = gr.Gallery(
                            label="Similar Images", columns=4, height=520, object_fit="cover"
                        )
                i_inputs = [i_query, i_k, i_thresh, i_date_from, i_date_to]
                i_btn.click(cb_image_search, i_inputs, [i_gallery, i_status])

            # ── Tab 3: Duplicate finder ────────────────────────────────────
            with gr.Tab("🔎 Find Duplicates"):
                gr.Markdown("Scan the index for near-duplicate images using cosine similarity.")
                with gr.Row():
                    d_thresh = gr.Slider(0.80, 1.0, value=0.97, step=0.01, label="Similarity Threshold")
                    d_btn = gr.Button("Scan for Duplicates", variant="primary")
                d_status = gr.Markdown()
                d_gallery = gr.Gallery(
                    label="Duplicate Images",
                    columns=4, height=380, object_fit="cover",
                )
                gr.Markdown(
                    "**Select files to delete** — extras are pre-checked (first image in each group is kept). "
                    "Review the selection before deleting."
                )
                d_checkbox = gr.CheckboxGroup(choices=[], label="", interactive=True)
                with gr.Row():
                    d_del_disk_btn = gr.Button("🗑️ Delete from Disk + Index", variant="stop")
                    d_del_idx_btn  = gr.Button("📦 Remove from Index Only", variant="secondary")
                d_del_status = gr.Markdown()

                d_btn.click(
                    cb_find_duplicates,
                    inputs=[d_thresh],
                    outputs=[d_gallery, d_status, d_checkbox],
                )
                d_del_disk_btn.click(
                    lambda paths: cb_delete_files(paths, True),
                    inputs=[d_checkbox],
                    outputs=[d_del_status],
                )
                d_del_idx_btn.click(
                    lambda paths: cb_delete_files(paths, False),
                    inputs=[d_checkbox],
                    outputs=[d_del_status],
                )

            # ── Tab 4: Index a folder ──────────────────────────────────────
            with gr.Tab("📁 Index Folder"):
                gr.Markdown("Add a folder to the index. Only new images are processed (incremental).")
                with gr.Row():
                    idx_path = gr.Textbox(
                        label="Folder Path",
                        placeholder="/home/user/Photos  or  C:\\Users\\User\\Pictures",
                    )
                    idx_btn = gr.Button("Index", variant="primary")
                idx_status = gr.Markdown()
                idx_btn.click(cb_index_folder, [idx_path], [idx_status])

            # ── Tab 5: Stats ───────────────────────────────────────────────
            with gr.Tab("📊 Stats"):
                stats_btn = gr.Button("Refresh Stats")
                stats_md = gr.Markdown(cb_stats())
                stats_btn.click(cb_stats, [], [stats_md])

    return demo


def launch(
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    config: ArchivistConfig | None = None,
) -> None:
    """Launch the Gradio app and block until the server is stopped."""
    from pathlib import Path
    app = create_app(config)
    # Allow Gradio to serve files from anywhere under the user's home directory
    # so photos stored in Pictures, Downloads, Desktop, etc. all display correctly
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        allowed_paths=[str(Path.home())],
    )
