"""Archivist-AI power features: duplicates, auto-tagging, organizer, watcher."""

from .duplicates import DuplicateFinder, DuplicateGroup
from .autotag import AutoTagger, DEFAULT_TAGS
from .organizer import Organizer, OrganizeResult
from .watcher import ArchivistWatcher

__all__ = [
    "DuplicateFinder",
    "DuplicateGroup",
    "AutoTagger",
    "DEFAULT_TAGS",
    "Organizer",
    "OrganizeResult",
    "ArchivistWatcher",
]
