"""
Archivist-AI CLI — powered by Typer + Rich.

Commands
────────
  archivist index   <dirs>   Index image directories (incremental)
  archivist search  <query>  Search by natural language
  archivist similar <image>  Reverse image search
  archivist dupes            Find near-duplicate images
  archivist tag              Auto-tag images in the index
  archivist copy   <query>   Search + copy results to a folder
  archivist watch  <dirs>    Watch directories for new images (auto-index)
  archivist clean            Remove stale entries for deleted files
  archivist stats            Show index statistics
  archivist ui               Launch the Gradio web UI
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from archivist import __version__
from archivist.config import ArchivistConfig
from archivist.core import (
    ArchivistIndexer,
    ArchivistSearcher,
    ArchivistStore,
    CLIPEmbedder,
)
from archivist.core.embedder import create_embedder

# ── App setup ─────────────────────────────────────────────────────────────────

app = typer.Typer(
    name="archivist",
    help="🗂️  Archivist-AI: Local-first AI image search & management.",
    rich_markup_mode="rich",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
console = Console()

# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_config(index_dir: Optional[Path] = None) -> ArchivistConfig:
    cfg = ArchivistConfig.load()
    if index_dir:
        cfg.index_dir = index_dir
    return cfg


def _open_store(config: ArchivistConfig) -> ArchivistStore:
    store = ArchivistStore(config.index_dir, embedding_dim=config.embedding_dim)
    store.open()
    return store


def _make_embedder(config: ArchivistConfig):
    """Return the right embedder — ONNX, SigLIP, or CLIP — based on config."""
    if config.use_onnx:
        from archivist.core.onnx_embedder import ONNXEmbedder
        from archivist.core.onnx_exporter import onnx_models_exist
        if not onnx_models_exist(config.onnx_dir):
            console.print(
                "[red]❌  ONNX models not found.[/red]\n"
                f"Run [bold]archivist export-onnx[/bold] first to generate them.\n"
                f"Expected in: {config.onnx_dir}"
            )
            raise typer.Exit(1)
        return ONNXEmbedder(onnx_dir=config.onnx_dir, batch_size=config.batch_size)

    return create_embedder(
        model_id=config.model_id,
        device=config.device,
        batch_size=config.batch_size,
        quantize=config.quantize,
    )


def _version_callback(value: bool):
    if value:
        console.print(f"Archivist-AI version [bold]{__version__}[/bold]")
        raise typer.Exit()


# ── Global options ────────────────────────────────────────────────────────────

@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-V", callback=_version_callback, is_eager=True, help="Show version"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
):
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")


# ── Commands ──────────────────────────────────────────────────────────────────

@app.command()
def index(
    directories: list[Path] = typer.Argument(
        ..., help="One or more directories to index", exists=True, file_okay=False, resolve_path=True
    ),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r/-R", help="Recurse into subdirectories"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Images per embedding batch"),
    index_dir: Optional[Path] = typer.Option(None, "--index-dir", "-i", help="Custom index directory"),
):
    """
    📁  Index image directories.

    Only newly added or changed images are processed (incremental).
    Subsequent runs on the same folder complete in seconds.

    Example:
        archivist index ~/Photos ~/Desktop/Screenshots
    """
    config = _load_config(index_dir)
    config.batch_size = batch_size

    embedder = _make_embedder(config)
    with _open_store(config) as store:
        indexer = ArchivistIndexer(store, embedder, batch_size=batch_size)
        result = indexer.index_multiple_directories(directories, recursive=recursive)

    console.print(Panel.fit(
        f"[green]✓  {result.newly_indexed} new images indexed[/green]\n"
        f"[dim]↩  {result.skipped_existing} already in index[/dim]\n"
        f"[dim]✗  {result.skipped_errors} errors[/dim]\n"
        f"[bold]Total in index: {result.newly_indexed + result.skipped_existing}[/bold]",
        title="[bold]Indexing complete[/bold]",
        border_style="green",
    ))


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural language query, e.g. 'people in suits'"),
    k: int = typer.Option(20, "--results", "-k", help="Max results to show"),
    threshold: float = typer.Option(0.0, "--threshold", "-t", help="Min cosine similarity (0–1)"),
    index_dir: Optional[Path] = typer.Option(None, "--index-dir", "-i"),
):
    """
    🔍  Search images by natural language description.

    Examples:
        archivist search "birthday party with balloons"
        archivist search "aerial view of city at night" -k 10 -t 0.25
    """
    config = _load_config(index_dir)
    embedder = _make_embedder(config)

    with _open_store(config) as store:
        if store.count() == 0:
            console.print("[red]Index is empty. Run [bold]archivist index <dir>[/bold] first.[/red]")
            raise typer.Exit(1)

        results = ArchivistSearcher(store, embedder).search_by_text(
            query, k=k, threshold=threshold
        )

    if not results:
        console.print(f"[yellow]No results for:[/yellow] '{query}'")
        return

    table = Table(title=f"Results: '{query}'", show_lines=True, expand=False)
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Score", style="bold green", width=7)
    table.add_column("File", style="cyan")
    table.add_column("Tags", style="dim")

    for r in results:
        table.add_row(
            str(r.rank),
            f"{r.similarity:.3f}",
            str(r.file_path),
            ", ".join(r.tags[:4]) if r.tags else "—",
        )

    console.print(table)
    console.print(f"\n[dim]{len(results)} results · threshold={threshold}[/dim]")


@app.command()
def similar(
    image_path: Path = typer.Argument(..., help="Query image path", exists=True, resolve_path=True),
    k: int = typer.Option(10, "--results", "-k", help="Max results"),
    threshold: float = typer.Option(0.0, "--threshold", "-t"),
    index_dir: Optional[Path] = typer.Option(None, "--index-dir", "-i"),
):
    """
    🖼️   Reverse image search — find visually similar images.

    Example:
        archivist similar ~/query_photo.jpg -k 15
    """
    config = _load_config(index_dir)
    embedder = _make_embedder(config)

    with _open_store(config) as store:
        if store.count() == 0:
            console.print("[red]Index is empty.[/red]")
            raise typer.Exit(1)

        results = ArchivistSearcher(store, embedder).search_by_image(
            image_path, k=k, threshold=threshold
        )

    if not results:
        console.print("[yellow]No similar images found.[/yellow]")
        return

    table = Table(title=f"Similar to: {image_path.name}", show_lines=True)
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Score", style="bold green", width=7)
    table.add_column("File", style="cyan")

    for r in results:
        table.add_row(str(r.rank), f"{r.similarity:.3f}", str(r.file_path))

    console.print(table)


@app.command()
def dupes(
    threshold: float = typer.Option(0.97, "--threshold", "-t", help="Similarity threshold (0–1)"),
    index_dir: Optional[Path] = typer.Option(None, "--index-dir", "-i"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save report to file"),
):
    """
    🔎  Find near-duplicate images in the index.

    Example:
        archivist dupes --threshold 0.99   # exact duplicates only
        archivist dupes -t 0.90 -o dupes_report.txt
    """
    from archivist.features.duplicates import DuplicateFinder

    config = _load_config(index_dir)

    with _open_store(config) as store:
        if store.count() < 2:
            console.print("[yellow]Need at least 2 images indexed to check for duplicates.[/yellow]")
            raise typer.Exit()

        with console.status(f"Scanning {store.count()} images for duplicates…"):
            groups = DuplicateFinder(store, threshold=threshold).find_duplicates()

    if not groups:
        console.print(f"[green]✓  No near-duplicates found at threshold {threshold}.[/green]")
        return

    total_images = sum(g.size for g in groups)
    console.print(
        Panel.fit(
            f"[bold red]{len(groups)} duplicate group(s)[/bold red]  "
            f"({total_images} images total)",
            border_style="red",
        )
    )

    lines = []
    for g in groups:
        console.print(
            f"\n[bold]Group {g.group_id + 1}[/bold]  "
            f"similarity={g.max_similarity:.4f}  "
            f"savings≈{g.potential_savings}"
        )
        for p in g.images:
            console.print(f"  [cyan]{p}[/cyan]")
            lines.append(str(p))

    if output:
        output.write_text("\n".join(lines))
        console.print(f"\n[dim]Report saved to {output}[/dim]")


@app.command()
def tag(
    index_dir: Optional[Path] = typer.Option(None, "--index-dir", "-i"),
    top_k: int = typer.Option(5, "--top-k", help="Max tags per image"),
    min_confidence: float = typer.Option(0.10, "--min-confidence", help="Min tag confidence"),
):
    """
    🏷️   Auto-tag all untagged images in the index using CLIP zero-shot classification.

    Example:
        archivist tag --top-k 8
    """
    from archivist.features.autotag import AutoTagger

    config = _load_config(index_dir)
    embedder = _make_embedder(config)

    with _open_store(config) as store:
        if store.count() == 0:
            console.print("[red]Index is empty.[/red]")
            raise typer.Exit(1)

        tagger = AutoTagger(embedder, store, top_k=top_k, min_confidence=min_confidence)
        tagged = tagger.tag_entire_index(show_progress=True)

    console.print(f"[green]✓  Tagged {tagged} images.[/green]")


@app.command()
def copy(
    query: str = typer.Argument(..., help="Search query"),
    destination: Path = typer.Argument(..., help="Destination directory"),
    k: int = typer.Option(50, "--results", "-k"),
    threshold: float = typer.Option(0.0, "--threshold", "-t"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing files"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Preview without copying"),
    index_dir: Optional[Path] = typer.Option(None, "--index-dir", "-i"),
):
    """
    📋  Search images and copy results to a folder.

    Example:
        archivist copy "beach sunset" ~/Organized/Sunsets --dry-run
        archivist copy "wedding photos" ~/Wedding -k 200
    """
    from archivist.features.organizer import Organizer

    config = _load_config(index_dir)
    embedder = _make_embedder(config)

    with _open_store(config) as store:
        if store.count() == 0:
            console.print("[red]Index is empty.[/red]")
            raise typer.Exit(1)

        results = ArchivistSearcher(store, embedder).search_by_text(
            query, k=k, threshold=threshold
        )

    if not results:
        console.print(f"[yellow]No results for: '{query}'[/yellow]")
        return

    console.print(f"Found [bold]{len(results)}[/bold] results for '{query}'")

    if dry_run:
        console.print("[yellow][dry-run] No files will be copied.[/yellow]")

    outcome = Organizer.copy_to(results, destination, overwrite=overwrite, dry_run=dry_run)

    console.print(Panel.fit(
        f"[green]Copied: {outcome.copied}[/green]\n"
        f"[dim]Skipped: {outcome.skipped}  ·  Errors: {outcome.errors}[/dim]\n"
        f"[dim]Destination: {destination}[/dim]",
        title="[bold]Copy complete[/bold]",
    ))


@app.command()
def watch(
    directories: list[Path] = typer.Argument(
        ..., help="Directories to watch", exists=True, file_okay=False, resolve_path=True
    ),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r/-R"),
    debounce: float = typer.Option(2.0, "--debounce", help="Seconds to wait after last event"),
    index_dir: Optional[Path] = typer.Option(None, "--index-dir", "-i"),
):
    """
    👁️   Watch directories for new images and auto-index them.

    Runs until Ctrl+C. New images are indexed within a few seconds of appearing.

    Example:
        archivist watch ~/Photos ~/Downloads --debounce 5
    """
    from archivist.features.watcher import ArchivistWatcher

    config = _load_config(index_dir)
    embedder = _make_embedder(config)

    with _open_store(config) as store:
        with ArchivistWatcher(
            store, embedder,
            watch_dirs=directories,
            recursive=recursive,
            debounce_seconds=debounce,
        ) as watcher:
            watcher.run_forever()


@app.command()
def clean(
    index_dir: Optional[Path] = typer.Option(None, "--index-dir", "-i"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """
    🧹  Remove index entries for files that no longer exist on disk.

    Run this after moving or deleting photos to keep the index in sync.
    """
    config = _load_config(index_dir)

    with _open_store(config) as store:
        if not yes:
            typer.confirm(
                f"Scan {store.count()} records for missing files?", abort=True
            )
        removed = store.remove_missing_files()

    if removed:
        console.print(f"[green]✓  Removed {removed} stale record(s).[/green]")
    else:
        console.print("[dim]Index is clean — no missing files found.[/dim]")


@app.command()
def stats(
    index_dir: Optional[Path] = typer.Option(None, "--index-dir", "-i"),
):
    """
    📊  Show index statistics.
    """
    config = _load_config(index_dir)

    with _open_store(config) as store:
        s = store.stats()

    console.print(Panel.fit(
        f"[bold]Total indexed:[/bold]   {s['total_indexed']}\n"
        f"[bold]FAISS vectors:[/bold]   {s['faiss_vectors']}\n"
        f"[bold]Embedding dim:[/bold]   {s['embedding_dim']}\n"
        f"[bold]DB size:[/bold]         {s['db_size_mb']} MB\n"
        f"[bold]Index file size:[/bold] {s['index_size_mb']} MB\n"
        f"[bold]Index location:[/bold]  {s['index_dir']}",
        title=f"[bold]Archivist-AI  v{__version__}[/bold]",
        border_style="blue",
    ))


@app.command("export-onnx")
def export_onnx(
    model_id: str = typer.Option(
        None, "--model", "-m",
        help="HuggingFace model id to export (defaults to config model_id)"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o",
        help="Where to write .onnx files (defaults to ~/.archivist/onnx)"
    ),
    opset: int = typer.Option(14, "--opset", help="ONNX opset version"),
    no_quantize: bool = typer.Option(False, "--no-quantize", help="Skip int8 quantization"),
):
    """
    ⚡  Export the embedding model to ONNX for 3-5x faster CPU inference.

    Exports the vision and text encoders to ONNX, then applies dynamic
    int8 quantization. This is a one-time operation (~60s). After export,
    enable ONNX by setting use_onnx=true in ~/.archivist/config.json.

    Example:
        archivist export-onnx
        archivist export-onnx --model google/siglip-base-patch16-224
    """
    from archivist.core.onnx_exporter import export_to_onnx, DEFAULT_ONNX_DIR

    config = _load_config()
    resolved_model = model_id or config.model_id
    resolved_dir = output_dir or config.onnx_dir or DEFAULT_ONNX_DIR

    console.print(Panel.fit(
        f"[bold]Model:[/bold]      {resolved_model}\n"
        f"[bold]Output:[/bold]     {resolved_dir}\n"
        f"[bold]Quantize:[/bold]   {'no' if no_quantize else 'yes (int8)'}",
        title="[bold]⚡  ONNX Export[/bold]",
        border_style="yellow",
    ))
    console.print("[dim]This takes ~60 seconds on first run…[/dim]\n")

    try:
        with console.status("Exporting to ONNX…"):
            out = export_to_onnx(
                model_id=resolved_model,
                output_dir=resolved_dir,
                opset=opset,
                quantize=not no_quantize,
            )

        console.print(Panel.fit(
            f"[green]✓  Models exported to:[/green]  {out}\n\n"
            "Enable ONNX in [bold]~/.archivist/config.json[/bold]:\n"
            '  [cyan]"use_onnx": true[/cyan]',
            title="[bold green]Export complete[/bold green]",
            border_style="green",
        ))
    except Exception as exc:
        console.print(f"[red]❌  Export failed: {exc}[/red]")
        raise typer.Exit(1)


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address"),
    port: int = typer.Option(7860, "--port", "-p", help="Port number"),
    share: bool = typer.Option(False, "--share", help="Create a public Gradio share link"),
    index_dir: Optional[Path] = typer.Option(None, "--index-dir", "-i"),
):
    """
    🌐  Launch the Gradio web UI.

    Opens at http://127.0.0.1:7860 by default.

    Example:
        archivist ui --port 8080
        archivist ui --share   # get a public URL for sharing
    """
    config = _load_config(index_dir)

    console.print(
        f"[bold]Launching Archivist-AI UI[/bold] at "
        f"[link=http://{host}:{port}]http://{host}:{port}[/link]"
    )
    console.print("[dim]Press Ctrl+C to stop.[/dim]\n")

    from archivist.ui.app import launch
    launch(host=host, port=port, share=share, config=config)


if __name__ == "__main__":
    app()
