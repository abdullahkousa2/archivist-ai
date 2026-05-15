"""
File management operations for Archivist-AI.

After searching for images, users typically want to act on the results:
copy them to a folder, move them, or rename them to reflect the search query.

All operations support dry_run=True for a safe preview before making changes.
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from archivist.core.store import SearchResult

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class OrganizeResult:
    """Summary of a file organisation operation."""
    copied: int = 0
    moved: int = 0
    renamed: int = 0
    errors: int = 0
    skipped: int = 0
    destination: Path | None = None

    def __str__(self) -> str:
        parts = []
        if self.copied:
            parts.append(f"{self.copied} copied")
        if self.moved:
            parts.append(f"{self.moved} moved")
        if self.renamed:
            parts.append(f"{self.renamed} renamed")
        if self.skipped:
            parts.append(f"{self.skipped} skipped")
        if self.errors:
            parts.append(f"{self.errors} errors")
        return ", ".join(parts) if parts else "nothing done"


# ── Organizer ─────────────────────────────────────────────────────────────────


class Organizer:
    """
    Performs bulk file operations on search results.

    All methods are static and side-effect-free except for the actual file
    system operations — no state is mutated on the store or embedder.

    Example
    -------
    >>> results = searcher.search_by_text("golden retriever", k=50)
    >>> outcome = Organizer.copy_to(results, Path("~/Dogs"), dry_run=False)
    >>> print(outcome)  # "42 copied, 8 skipped"
    """

    @staticmethod
    def copy_to(
        results: list[SearchResult],
        destination: Path | str,
        overwrite: bool = False,
        dry_run: bool = False,
    ) -> OrganizeResult:
        """
        Copy search results to a destination folder.

        Parameters
        ----------
        results     : Output of ArchivistSearcher.search_by_text / search_by_image.
        destination : Target directory (created if it doesn't exist).
        overwrite   : If False, skip files that already exist at destination.
        dry_run     : Log what would happen without actually copying.

        Returns
        -------
        OrganizeResult with operation counts.
        """
        destination = Path(destination).expanduser()
        outcome = OrganizeResult(destination=destination)

        if not dry_run:
            destination.mkdir(parents=True, exist_ok=True)

        for result in results:
            src = result.file_path

            if not src.exists():
                logger.warning(f"Source missing: {src}")
                outcome.errors += 1
                continue

            dst = destination / src.name

            # Handle filename collisions
            if dst.exists():
                if overwrite:
                    pass  # Will overwrite
                else:
                    logger.debug(f"Skipping existing: {dst.name}")
                    outcome.skipped += 1
                    continue

            if dry_run:
                logger.info(f"[dry-run] Would copy: {src.name} → {destination}/")
                outcome.copied += 1
                continue

            try:
                shutil.copy2(src, dst)
                outcome.copied += 1
                logger.debug(f"Copied: {src.name}")
            except (OSError, shutil.Error) as exc:
                logger.error(f"Copy failed for {src.name}: {exc}")
                outcome.errors += 1

        return outcome

    @staticmethod
    def move_to(
        results: list[SearchResult],
        destination: Path | str,
        overwrite: bool = False,
        dry_run: bool = False,
    ) -> OrganizeResult:
        """
        Move search results to a destination folder.

        Note: Moving files updates their location on disk but does NOT update
        the Archivist index. Run `archivist clean` after moving to remove stale
        entries, then re-index the destination folder.
        """
        destination = Path(destination).expanduser()
        outcome = OrganizeResult(destination=destination)

        if not dry_run:
            destination.mkdir(parents=True, exist_ok=True)

        for result in results:
            src = result.file_path

            if not src.exists():
                logger.warning(f"Source missing: {src}")
                outcome.errors += 1
                continue

            dst = destination / src.name

            if dst.exists() and not overwrite:
                logger.debug(f"Skipping existing: {dst.name}")
                outcome.skipped += 1
                continue

            if dry_run:
                logger.info(f"[dry-run] Would move: {src.name} → {destination}/")
                outcome.moved += 1
                continue

            try:
                shutil.move(str(src), str(dst))
                outcome.moved += 1
                logger.debug(f"Moved: {src.name}")
            except (OSError, shutil.Error) as exc:
                logger.error(f"Move failed for {src.name}: {exc}")
                outcome.errors += 1

        return outcome

    @staticmethod
    def rename_with_query(
        results: list[SearchResult],
        query: str,
        prefix_rank: bool = True,
        dry_run: bool = False,
    ) -> OrganizeResult:
        """
        Rename result files to embed the search query and rank.

        Example output: "001_sunset_beach_IMG_4321.jpg"

        Parameters
        ----------
        results      : Search results to rename.
        query        : The query string used for naming.
        prefix_rank  : If True, prepend zero-padded rank number.
        dry_run      : Preview without renaming.

        Note: Like move_to, renaming does not update the index. Run
        `archivist clean && archivist index <dir>` to re-sync.
        """
        outcome = OrganizeResult()
        safe_query = _sanitize_for_filename(query)

        for result in results:
            src = result.file_path
            if not src.exists():
                outcome.errors += 1
                continue

            rank_prefix = f"{result.rank:03d}_" if prefix_rank else ""
            new_name = f"{rank_prefix}{safe_query}_{src.stem}{src.suffix}"
            dst = src.parent / new_name

            if dry_run:
                logger.info(f"[dry-run] Would rename: {src.name} → {new_name}")
                outcome.renamed += 1
                continue

            if dst.exists():
                logger.debug(f"Skipping: destination already exists {dst.name}")
                outcome.skipped += 1
                continue

            try:
                src.rename(dst)
                outcome.renamed += 1
                logger.debug(f"Renamed: {src.name} → {new_name}")
            except OSError as exc:
                logger.error(f"Rename failed for {src.name}: {exc}")
                outcome.errors += 1

        return outcome


# ── Helpers ───────────────────────────────────────────────────────────────────


def _sanitize_for_filename(s: str, max_len: int = 40) -> str:
    """Convert a string to a safe filename component."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)       # Remove special chars
    s = re.sub(r"[\s_-]+", "_", s)        # Collapse whitespace/dashes
    return s[:max_len].strip("_")
