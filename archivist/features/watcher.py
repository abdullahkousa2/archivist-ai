"""
Filesystem watcher for automatic re-indexing in Archivist-AI.

Watches one or more directories for newly added or renamed image files and
automatically indexes them in the background. Built on watchdog for
cross-platform filesystem event support (inotify on Linux, FSEvents on macOS,
ReadDirectoryChangesW on Windows).

Debouncing
──────────
New events are buffered and only processed after a quiet period (default 2 s).
This prevents thrashing when many files are added at once (e.g., photo import).

Usage
─────
    with ArchivistWatcher(store, embedder, [Path("~/Photos")]) as watcher:
        watcher.run_forever()   # Block until Ctrl+C
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Event

from watchdog.events import (
    FileCreatedEvent,
    FileMovedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from archivist.core.embedder import BaseEmbedder
from archivist.core.indexer import SUPPORTED_EXTENSIONS, ArchivistIndexer
from archivist.core.store import ArchivistStore

logger = logging.getLogger(__name__)


# ── Watchdog event handler ────────────────────────────────────────────────────


class _ImageEventHandler(FileSystemEventHandler):
    """Routes filesystem events for supported image types into a Queue."""

    def __init__(self, queue: Queue) -> None:
        self.queue = queue

    def on_created(self, event: FileCreatedEvent) -> None:
        if not event.is_directory:
            path = Path(event.src_path)
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                self.queue.put(path)
                logger.debug(f"New image detected: {path.name}")

    def on_moved(self, event: FileMovedEvent) -> None:
        """Handle renames / moves into a watched directory."""
        if not event.is_directory:
            path = Path(event.dest_path)
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                self.queue.put(path)
                logger.debug(f"Image moved/renamed: {path.name}")


# ── Watcher ───────────────────────────────────────────────────────────────────


class ArchivistWatcher:
    """
    Background watcher that auto-indexes new images as they appear.

    Parameters
    ----------
    store            : Open ArchivistStore instance.
    embedder         : Shared CLIPEmbedder.
    watch_dirs       : Directories to monitor.
    recursive        : Watch subdirectories too.
    debounce_seconds : Wait this long after the last event before processing.

    Example
    -------
    >>> with ArchivistWatcher(store, embedder, [Path("~/Photos")]) as w:
    ...     w.run_forever()
    Watching: /home/user/Photos
    Press Ctrl+C to stop.
    [12:34:56] Auto-indexing 3 new image(s)...
    """

    def __init__(
        self,
        store: ArchivistStore,
        embedder: BaseEmbedder,
        watch_dirs: list[Path | str],
        recursive: bool = True,
        debounce_seconds: float = 2.0,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.watch_dirs = [Path(d).expanduser().resolve() for d in watch_dirs]
        self.recursive = recursive
        self.debounce_seconds = debounce_seconds

        self._queue: Queue[Path] = Queue()
        self._observer = Observer()
        self._stop_event = Event()
        self._indexer = ArchivistIndexer(store, embedder, batch_size=16)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the filesystem observer."""
        handler = _ImageEventHandler(self._queue)
        for directory in self.watch_dirs:
            if not directory.is_dir():
                logger.warning(f"Watch directory not found: {directory}")
                continue
            self._observer.schedule(handler, str(directory), recursive=self.recursive)
            logger.info(f"Watching: {directory}")
        self._observer.start()

    def stop(self) -> None:
        """Signal the watcher to stop and wait for the observer thread."""
        self._stop_event.set()
        self._observer.stop()
        self._observer.join()

    def __enter__(self) -> "ArchivistWatcher":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run_forever(self) -> None:
        """
        Block and process incoming image events until Ctrl+C or stop() is called.

        New files are batched with debounce to avoid redundant indexing
        when a burst of images is added at once.
        """
        logger.info(
            f"Watcher running  |  monitoring {len(self.watch_dirs)} director(ies)."
        )
        logger.info("Press Ctrl+C to stop.")

        pending: list[Path] = []
        last_event_time = 0.0

        try:
            while not self._stop_event.is_set():
                # Drain the queue
                try:
                    path = self._queue.get(timeout=0.5)
                    pending.append(path)
                    last_event_time = time.monotonic()
                except Empty:
                    pass

                # Flush after debounce quiet period
                elapsed = time.monotonic() - last_event_time
                if pending and elapsed >= self.debounce_seconds:
                    self._process_batch(pending)
                    pending.clear()

        except KeyboardInterrupt:
            logger.info("Watcher stopped by user.")
        finally:
            # Process any remaining events before exit
            if pending:
                self._process_batch(pending)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _process_batch(self, paths: list[Path]) -> None:
        """Index the unique parent directories of a batch of new files."""
        unique_dirs = {p.parent for p in paths}
        count = len(paths)
        logger.info(f"Auto-indexing {count} new image(s) from {len(unique_dirs)} director(ies)...")

        result = self._indexer.index_multiple_directories(
            list(unique_dirs),
            recursive=False,
            show_progress=False,
        )

        logger.info(
            f"Done  |  {result.newly_indexed} new, "
            f"{result.skipped_existing} already indexed, "
            f"{result.skipped_errors} errors"
        )
