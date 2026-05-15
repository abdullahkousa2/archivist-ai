"""
Image and text embedding engine for Archivist-AI.

Backends
--------
- CLIPEmbedder     : openai/clip-vit-base-patch32 via HuggingFace Transformers.
                     Dynamic int8 quantization for 2-3x faster CPU inference.
- SigLIPEmbedder   : google/siglip-base-patch16-224.  Significantly better
                     accuracy than CLIP ViT-B/32 at the same model size.
                     Uses sigmoid loss training — handles descriptive queries
                     ("food on a table") much better than CLIP.
- create_embedder  : Factory that auto-selects backend from model_id string.

All embedders share the BaseEmbedder interface so the rest of the codebase
is backend-agnostic — swap models without touching any other module.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


# ── Abstract base ─────────────────────────────────────────────────────────────


class BaseEmbedder(ABC):
    """
    Abstract embedding backend.

    All embeddings are L2-normalised float32 arrays of shape (N, EMBEDDING_DIM),
    so cosine similarity reduces to inner product — exploited by FAISS IndexFlatIP.
    """

    EMBEDDING_DIM: int = 512

    @abstractmethod
    def encode_text(self, texts: list[str]) -> np.ndarray:
        """Encode text strings → L2-normalised array (N, dim)."""
        ...

    @abstractmethod
    def encode_images(
        self, image_paths: list[Path]
    ) -> tuple[np.ndarray, list[Path]]:
        """
        Encode image files → (embeddings, successful_paths).

        Images that cannot be opened are skipped gracefully; the returned
        successful_paths list tells you which files were actually embedded.
        """
        ...

    @abstractmethod
    def encode_pil_images(self, images: list[Image.Image]) -> np.ndarray:
        """Encode pre-loaded PIL Images → L2-normalised array (N, dim)."""
        ...


# ── CLIP implementation ───────────────────────────────────────────────────────


class CLIPEmbedder(BaseEmbedder):
    """
    CLIP ViT-B/32 embedder optimised for CPU-only inference.

    Features
    --------
    - Lazy model loading: the model is downloaded and initialised only on
      the first call to encode_text / encode_images.
    - Dynamic int8 quantization (torch.quantization.quantize_dynamic) applied
      to all Linear layers, giving 2-3x speedup on CPU with negligible accuracy
      loss on retrieval tasks.
    - Batch processing: large lists are chunked to avoid OOM on small machines.
    - Graceful error handling: unreadable images are skipped with a warning.

    Models are cached by HuggingFace in ~/.cache/huggingface/ after the first
    download (~330 MB for ViT-B/32).
    """

    EMBEDDING_DIM = 512
    DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"

    def __init__(
        self,
        model_id: str | None = None,
        device: str = "cpu",
        batch_size: int = 32,
        quantize: bool = True,
    ) -> None:
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self.device = device
        self.batch_size = batch_size
        self._quantize = quantize

        # Lazy-loaded
        self._model = None
        self._processor = None

    # ── Public API ────────────────────────────────────────────────────────────

    def encode_text(self, texts: list[str]) -> np.ndarray:
        """
        Encode text strings into L2-normalised embeddings.

        Returns
        -------
        np.ndarray of shape (len(texts), 512), dtype float32.
        """
        self._ensure_loaded()
        import torch

        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            inputs = self._processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(self.device)

            with torch.no_grad():
                # Use text_model directly — more robust across transformers versions
                # than get_text_features() which returns different types in newer releases
                text_out = self._model.text_model(**inputs, return_dict=True)
                pooled = text_out.pooler_output          # (N, hidden_dim)
                emb = self._model.text_projection(pooled)  # (N, 512)

            emb = self._l2_normalize(emb)
            all_embeddings.append(emb.cpu().float().numpy())

        return np.vstack(all_embeddings) if all_embeddings else np.empty((0, self.EMBEDDING_DIM), dtype=np.float32)

    def encode_images(
        self, image_paths: list[Path]
    ) -> tuple[np.ndarray, list[Path]]:
        """
        Encode image files into L2-normalised embeddings.

        Unreadable / corrupt files are silently skipped. The caller can compare
        len(successful_paths) against len(image_paths) to detect failures.

        Returns
        -------
        (embeddings, successful_paths) where embeddings has shape
        (len(successful_paths), 512).
        """
        self._ensure_loaded()

        all_embeddings: list[np.ndarray] = []
        successful_paths: list[Path] = []

        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i : i + self.batch_size]
            pil_images: list[Image.Image] = []
            valid_paths: list[Path] = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    pil_images.append(img)
                    valid_paths.append(path)
                except (UnidentifiedImageError, OSError, Exception) as exc:
                    logger.warning(f"Skipping unreadable image {path.name}: {exc}")

            if not pil_images:
                continue

            batch_emb = self._encode_pil_batch(pil_images)
            all_embeddings.append(batch_emb)
            successful_paths.extend(valid_paths)

        if not all_embeddings:
            return np.empty((0, self.EMBEDDING_DIM), dtype=np.float32), []

        return np.vstack(all_embeddings), successful_paths

    def encode_pil_images(self, images: list[Image.Image]) -> np.ndarray:
        """Encode pre-loaded PIL Images."""
        self._ensure_loaded()
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]
            all_embeddings.append(self._encode_pil_batch(batch))
        return np.vstack(all_embeddings) if all_embeddings else np.empty((0, self.EMBEDDING_DIM), dtype=np.float32)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Lazy-load the CLIP model on first use."""
        if self._model is not None:
            return

        logger.info(f"Loading CLIP model '{self.model_id}' (first run may download ~330 MB)...")

        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._processor = CLIPProcessor.from_pretrained(self.model_id)
        model = CLIPModel.from_pretrained(self.model_id)
        model.eval()

        if self._quantize and self.device == "cpu":
            logger.info("Applying dynamic int8 quantization for faster CPU inference...")
            model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )

        self._model = model.to(self.device)
        logger.info("CLIP model ready ✓")

    def _encode_pil_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Encode a batch of PIL Images. Model must already be loaded."""
        import torch

        inputs = self._processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # Use vision_model directly — avoids get_image_features() returning
            # BaseModelOutputWithPooling in newer transformers versions
            vision_out = self._model.vision_model(**inputs, return_dict=True)
            pooled = vision_out.pooler_output               # (N, hidden_dim)
            emb = self._model.visual_projection(pooled)     # (N, 512)
        return self._l2_normalize(emb).cpu().float().numpy()

    @staticmethod
    def _l2_normalize(tensor):
        """L2-normalise along the last dimension (in-place safe version)."""
        import torch
        return tensor / tensor.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        loaded = self._model is not None
        return (
            f"CLIPEmbedder("
            f"model={self.model_id!r}, "
            f"device={self.device!r}, "
            f"quantize={self._quantize}, "
            f"loaded={loaded})"
        )


# ── SigLIP implementation ─────────────────────────────────────────────────────


class SigLIPEmbedder(BaseEmbedder):
    """
    SigLIP ViT-B/16 embedder — better accuracy than CLIP at the same model size.

    SigLIP (Sigmoid Loss for Language Image Pre-Training) uses a per-sample
    sigmoid loss instead of CLIP's softmax, which results in:
    - Stronger open-vocabulary recognition (food, objects, scenes)
    - Better handling of descriptive / compositional queries
    - 768-dimensional embedding space (vs CLIP's 512)

    Default model: google/siglip-base-patch16-224  (~370 MB, downloaded once)

    Features
    --------
    - Lazy model loading
    - Dynamic int8 quantization (Linear layers) for fast CPU inference
    - Batch processing with graceful error handling
    - L2-normalised embeddings → cosine similarity = inner product
    """

    EMBEDDING_DIM = 768
    DEFAULT_MODEL_ID = "google/siglip-base-patch16-224"

    def __init__(
        self,
        model_id: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        quantize: bool = True,
    ) -> None:
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self.device = device
        self.batch_size = batch_size
        self._quantize = quantize
        self._model = None
        self._processor = None

    # ── Public API ────────────────────────────────────────────────────────────

    def encode_text(self, texts: list[str]) -> np.ndarray:
        """Encode text strings into L2-normalised 768-dim embeddings."""
        self._ensure_loaded()
        import torch

        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # SigLIP processor pads to max_length=64 by default
            inputs = self._processor(
                text=batch,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                # Use text_model directly — get_text_features() returns
                # BaseModelOutputWithPooling in newer transformers versions.
                # SigLIP processor may omit attention_mask (pads to fixed
                # max_length=64, so all positions are valid by default).
                input_ids = inputs["input_ids"]
                attention_mask = inputs.get("attention_mask")
                if attention_mask is None:
                    import torch
                    attention_mask = torch.ones_like(input_ids)
                text_out = self._model.text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                emb = text_out.pooler_output  # (N, 768)

            emb = self._l2_normalize(emb)
            all_embeddings.append(emb.cpu().float().numpy())

        return (
            np.vstack(all_embeddings)
            if all_embeddings
            else np.empty((0, self.EMBEDDING_DIM), dtype=np.float32)
        )

    def encode_images(
        self, image_paths: list[Path]
    ) -> tuple[np.ndarray, list[Path]]:
        """Encode image files → (L2-normalised embeddings, successful_paths)."""
        self._ensure_loaded()

        all_embeddings: list[np.ndarray] = []
        successful_paths: list[Path] = []

        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i : i + self.batch_size]
            pil_images: list[Image.Image] = []
            valid_paths: list[Path] = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    pil_images.append(img)
                    valid_paths.append(path)
                except (UnidentifiedImageError, OSError, Exception) as exc:
                    logger.warning(f"Skipping unreadable image {path.name}: {exc}")

            if not pil_images:
                continue

            batch_emb = self._encode_pil_batch(pil_images)
            all_embeddings.append(batch_emb)
            successful_paths.extend(valid_paths)

        if not all_embeddings:
            return np.empty((0, self.EMBEDDING_DIM), dtype=np.float32), []

        return np.vstack(all_embeddings), successful_paths

    def encode_pil_images(self, images: list[Image.Image]) -> np.ndarray:
        """Encode pre-loaded PIL Images."""
        self._ensure_loaded()
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]
            all_embeddings.append(self._encode_pil_batch(batch))
        return (
            np.vstack(all_embeddings)
            if all_embeddings
            else np.empty((0, self.EMBEDDING_DIM), dtype=np.float32)
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Lazy-load the SigLIP model on first use."""
        if self._model is not None:
            return

        logger.info(
            f"Loading SigLIP model '{self.model_id}' "
            "(first run may download ~370 MB)…"
        )

        import torch
        from transformers import SiglipModel, SiglipProcessor

        self._processor = SiglipProcessor.from_pretrained(self.model_id)
        model = SiglipModel.from_pretrained(self.model_id)
        model.eval()

        if self._quantize and self.device == "cpu":
            logger.info("Applying dynamic int8 quantization for faster CPU inference…")
            model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )

        self._model = model.to(self.device)
        logger.info("SigLIP model ready ✓")

    def _encode_pil_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Encode a batch of PIL Images. Model must be loaded."""
        import torch

        inputs = self._processor(
            images=images, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            # Use vision_model directly — get_image_features() returns
            # BaseModelOutputWithPooling in newer transformers versions
            vision_out = self._model.vision_model(
                pixel_values=inputs["pixel_values"],
                return_dict=True,
            )
            emb = vision_out.pooler_output  # (N, 768)
        return self._l2_normalize(emb).cpu().float().numpy()

    @staticmethod
    def _l2_normalize(tensor):
        """L2-normalise along the last dimension."""
        import torch
        return tensor / tensor.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    def __repr__(self) -> str:
        loaded = self._model is not None
        return (
            f"SigLIPEmbedder("
            f"model={self.model_id!r}, "
            f"device={self.device!r}, "
            f"quantize={self._quantize}, "
            f"loaded={loaded})"
        )


# ── Factory ───────────────────────────────────────────────────────────────────


def create_embedder(
    model_id: str,
    device: str = "cpu",
    batch_size: int | None = None,
    quantize: bool = True,
) -> BaseEmbedder:
    """
    Auto-select the right embedder backend from a model_id string.

    - model_id contains "siglip" → SigLIPEmbedder (768-dim)
    - anything else              → CLIPEmbedder    (512-dim)

    Examples
    --------
    >>> emb = create_embedder("google/siglip-base-patch16-224")
    >>> emb = create_embedder("openai/clip-vit-base-patch32")
    """
    model_id_lower = model_id.lower()

    if "siglip" in model_id_lower:
        return SigLIPEmbedder(
            model_id=model_id,
            device=device,
            batch_size=batch_size or 16,
            quantize=quantize,
        )
    else:
        return CLIPEmbedder(
            model_id=model_id,
            device=device,
            batch_size=batch_size or 32,
            quantize=quantize,
        )
