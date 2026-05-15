"""
Zero-shot auto-tagging for Archivist-AI using CLIP.

How it works
────────────
CLIP was trained to align image and text embeddings. We exploit this by:
1. Pre-computing text embeddings for a vocabulary of descriptive tags, e.g.
   "a photo of a dog", "a photo of a sunset".
2. For each image, computing the cosine similarity against all tag embeddings.
3. Applying softmax to convert similarities to probabilities.
4. Storing the top-K tags (above a confidence threshold) in SQLite.

This gives us tag-based browsing ("show me all 'outdoor' 'sunset' photos")
for free, with no additional model training or API calls.

Tag vocabulary
──────────────
The default vocabulary covers common scenes, subjects, events, and moods.
Pass a custom `tags` list to AutoTagger to use your own labels.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from archivist.core.embedder import BaseEmbedder
from archivist.core.store import ArchivistStore

logger = logging.getLogger(__name__)


# ── Default vocabulary ────────────────────────────────────────────────────────

DEFAULT_TAGS: list[str] = [
    # Scene / setting
    "indoor", "outdoor", "nature", "urban", "landscape", "abstract",
    # Lighting / weather / time
    "sunny", "cloudy", "night", "sunset", "sunrise", "golden hour", "snow", "rain",
    # People
    "people", "portrait", "group photo", "selfie", "children", "baby", "elderly",
    # Animals
    "dog", "cat", "bird", "horse", "wildlife", "insects",
    # Activities
    "sport", "running", "swimming", "cooking", "dining", "reading", "music",
    # Places
    "beach", "mountains", "forest", "ocean", "lake", "river", "desert",
    "city", "street", "park", "garden",
    # Events
    "party", "wedding", "birthday", "graduation", "holiday", "concert", "festival",
    # Objects / subjects
    "food", "drink", "coffee", "architecture", "car", "technology", "flowers",
    "plants", "art", "fashion",
    # Mood / style
    "colorful", "black and white", "minimalist", "vintage", "aerial",
]


# ── AutoTagger ────────────────────────────────────────────────────────────────


class AutoTagger:
    """
    Assigns descriptive tags to images via CLIP zero-shot classification.

    Tag text embeddings are computed once and cached — subsequent calls
    do not repeat that computation.

    Example
    -------
    >>> tagger = AutoTagger(embedder, store)
    >>> tags = tagger.tag_image_path(Path("beach_sunset.jpg"))
    >>> print(tags)  # [("sunset", 0.42), ("beach", 0.38), ("outdoor", 0.31)]
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        store: ArchivistStore,
        tags: list[str] | None = None,
        top_k: int = 5,
        min_confidence: float = 0.10,
        temperature: float = 100.0,
    ) -> None:
        """
        Parameters
        ----------
        embedder        : Shared CLIP embedder instance.
        store           : Archivist store (for updating tags in SQLite).
        tags            : Custom tag vocabulary; defaults to DEFAULT_TAGS.
        top_k           : Maximum tags per image.
        min_confidence  : Minimum softmax probability to assign a tag.
        temperature     : Softmax temperature scaling (higher = sharper).
        """
        self.embedder = embedder
        self.store = store
        self.tags = tags or DEFAULT_TAGS
        self.top_k = top_k
        self.min_confidence = min_confidence
        self.temperature = temperature
        self._tag_embeddings: np.ndarray | None = None  # Lazy-computed cache

    # ── Public API ────────────────────────────────────────────────────────────

    def tag_image(self, image: Image.Image) -> list[tuple[str, float]]:
        """
        Tag a PIL Image.

        Returns
        -------
        List of (tag, confidence) tuples sorted by confidence descending.
        Only tags above min_confidence are returned, up to top_k.
        """
        tag_embs = self._get_tag_embeddings()
        img_emb = self.embedder.encode_pil_images([image])[0]  # (512,)

        # Cosine similarity (dot product, since both are L2-normalised)
        similarities = tag_embs @ img_emb  # (N_tags,)

        # Softmax with temperature scaling
        scaled = similarities * self.temperature
        exp_scaled = np.exp(scaled - scaled.max())  # Stable softmax
        probs = exp_scaled / exp_scaled.sum()

        # Select top-K above threshold
        top_indices = np.argsort(probs)[::-1][: self.top_k]
        return [
            (self.tags[i], float(probs[i]))
            for i in top_indices
            if float(probs[i]) >= self.min_confidence
        ]

    def tag_image_path(self, image_path: Path) -> list[tuple[str, float]]:
        """Tag an image from a file path."""
        img = Image.open(image_path).convert("RGB")
        return self.tag_image(img)

    def tag_and_store(self, faiss_id: int, image: Image.Image, replace: bool = True) -> list[tuple[str, float]]:
        """Tag an image and write the tags to SQLite."""
        tags = self.tag_image(image)
        if tags:
            self.store.update_tags(faiss_id, tags, replace=replace)
        return tags

    def tag_entire_index(self, show_progress: bool = True) -> int:
        """
        Tag every image in the index that has no tags yet.

        Returns
        -------
        Number of images tagged.
        """
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

        rows = self.store._db.execute("""
            SELECT i.faiss_id, i.file_path
            FROM   images i
            LEFT JOIN tags t ON t.image_id = i.faiss_id
            WHERE  t.image_id IS NULL
        """).fetchall()

        if not rows:
            logger.info("All images already have tags.")
            return 0

        tagged = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            disable=not show_progress,
        ) as progress:
            task = progress.add_task(f"Auto-tagging {len(rows)} images...", total=len(rows))

            for row in rows:
                try:
                    img = Image.open(row["file_path"]).convert("RGB")
                    self.tag_and_store(row["faiss_id"], img)
                    tagged += 1
                except Exception as exc:
                    logger.warning(f"Could not tag {row['file_path']}: {exc}")
                finally:
                    progress.advance(task)

        logger.info(f"Tagged {tagged} images.")
        return tagged

    # ── Internals ─────────────────────────────────────────────────────────────

    def _get_tag_embeddings(self) -> np.ndarray:
        """
        Compute and cache tag text embeddings.

        Uses CLIP's "a photo of {tag}" prompt template, which consistently
        outperforms bare tag labels in zero-shot classification.
        """
        if self._tag_embeddings is None:
            prompts = [f"a photo of {tag}" for tag in self.tags]
            logger.debug(f"Computing embeddings for {len(prompts)} tags...")
            self._tag_embeddings = self.embedder.encode_text(prompts)
        return self._tag_embeddings
