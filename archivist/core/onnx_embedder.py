"""
ONNX Runtime embedding backend for Archivist-AI.

Uses pre-exported ONNX models (vision + text encoders) via onnxruntime
for 3-5x faster CPU inference compared to the PyTorch/transformers backend.

Usage
-----
    # First export the model (one-time, ~60s):
    archivist export-onnx

    # Then enable in config:
    # ~/.archivist/config.json → "use_onnx": true

    # Or use programmatically:
    from archivist.core.onnx_embedder import ONNXEmbedder
    emb = ONNXEmbedder(onnx_dir=Path.home() / ".archivist" / "onnx")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, UnidentifiedImageError

from .embedder import BaseEmbedder

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class ONNXEmbedder(BaseEmbedder):
    """
    ONNX Runtime embedding engine — 3-5x faster than PyTorch on CPU.

    Loads pre-exported vision_encoder[_quant].onnx and
    text_encoder[_quant].onnx from onnx_dir and runs inference
    through onnxruntime.InferenceSession.

    The processor (tokenizer + image preprocessor) is still loaded from
    HuggingFace (no internet needed after first download).

    Embedding dim is read from the ONNX model itself, so this class
    works with any model exported via onnx_exporter.py.
    """

    def __init__(
        self,
        onnx_dir: PathLike,
        batch_size: int = 32,
        num_threads: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        onnx_dir:    Directory containing the exported .onnx files and
                     processor_id.txt (written by export_to_onnx()).
        batch_size:  Images / texts per inference batch.
        num_threads: onnxruntime intra-op threads (None = auto).
        """
        self.onnx_dir = Path(onnx_dir)
        self.batch_size = batch_size
        self.num_threads = num_threads

        self._vision_session = None
        self._text_session = None
        self._processor = None
        self._embedding_dim: int | None = None

    # ── BaseEmbedder interface ────────────────────────────────────────────────

    @property
    def EMBEDDING_DIM(self) -> int:  # type: ignore[override]
        self._ensure_loaded()
        return self._embedding_dim  # type: ignore[return-value]

    def encode_text(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            inputs = self._processor(
                text=batch,
                return_tensors="np",
                padding="max_length",
                truncation=True,
            )
            # SigLIP processor may omit attention_mask (pads to fixed
            # max_length so all positions are valid — fill with ones).
            input_ids = inputs["input_ids"].astype(np.int64)
            raw_mask = inputs.get("attention_mask")
            attention_mask = (
                raw_mask.astype(np.int64)
                if raw_mask is not None
                else np.ones_like(input_ids)
            )
            out = self._text_session.run(
                ["text_features"],
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                },
            )
            emb = out[0]  # (N, dim)
            all_embeddings.append(self._l2_normalize(emb))

        return (
            np.vstack(all_embeddings)
            if all_embeddings
            else np.empty((0, self._embedding_dim), dtype=np.float32)
        )

    def encode_images(
        self, image_paths: list[Path]
    ) -> tuple[np.ndarray, list[Path]]:
        self._ensure_loaded()

        all_embeddings: list[np.ndarray] = []
        successful_paths: list[Path] = []

        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i : i + self.batch_size]
            pil_images: list[Image.Image] = []
            valid_paths: list[Path] = []

            for path in batch_paths:
                try:
                    pil_images.append(Image.open(path).convert("RGB"))
                    valid_paths.append(path)
                except (UnidentifiedImageError, OSError, Exception) as exc:
                    logger.warning(f"Skipping {path.name}: {exc}")

            if not pil_images:
                continue

            emb = self._encode_pil_batch(pil_images)
            all_embeddings.append(emb)
            successful_paths.extend(valid_paths)

        if not all_embeddings:
            return np.empty((0, self._embedding_dim), dtype=np.float32), []

        return np.vstack(all_embeddings), successful_paths

    def encode_pil_images(self, images: list[Image.Image]) -> np.ndarray:
        self._ensure_loaded()
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]
            all_embeddings.append(self._encode_pil_batch(batch))
        return (
            np.vstack(all_embeddings)
            if all_embeddings
            else np.empty((0, self._embedding_dim), dtype=np.float32)
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._vision_session is not None:
            return

        import onnxruntime as ort

        # Prefer quantized models if available
        vision_path = self._pick(
            self.onnx_dir / "vision_encoder_quant.onnx",
            self.onnx_dir / "vision_encoder.onnx",
        )
        text_path = self._pick(
            self.onnx_dir / "text_encoder_quant.onnx",
            self.onnx_dir / "text_encoder.onnx",
        )

        logger.info(f"Loading ONNX sessions from {self.onnx_dir}…")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.num_threads is not None:
            opts.intra_op_num_threads = self.num_threads

        self._vision_session = ort.InferenceSession(
            str(vision_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._text_session = ort.InferenceSession(
            str(text_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )

        # Read embedding dim from the vision model output shape
        out_info = self._vision_session.get_outputs()[0]
        self._embedding_dim = out_info.shape[-1]

        # Load processor (tokenizer + image preprocessor)
        processor_id_path = self.onnx_dir / "processor_id.txt"
        if processor_id_path.exists():
            processor_id = processor_id_path.read_text().strip()
        else:
            processor_id = "google/siglip-base-patch16-224"
            logger.warning(
                f"processor_id.txt not found in {self.onnx_dir}; "
                f"defaulting to {processor_id}"
            )

        if "siglip" in processor_id.lower():
            from transformers import SiglipProcessor
            self._processor = SiglipProcessor.from_pretrained(processor_id)
        else:
            from transformers import CLIPProcessor
            self._processor = CLIPProcessor.from_pretrained(processor_id)

        logger.info(
            f"ONNX engine ready ✓  (dim={self._embedding_dim}, "
            f"vision={vision_path.name}, text={text_path.name})"
        )

    def _encode_pil_batch(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self._processor(images=images, return_tensors="np")
        out = self._vision_session.run(
            ["image_features"],
            {"pixel_values": inputs["pixel_values"].astype(np.float32)},
        )
        return self._l2_normalize(out[0])

    @staticmethod
    def _pick(*paths: Path) -> Path:
        """Return the first path that exists."""
        for p in paths:
            if p.exists():
                return p
        raise FileNotFoundError(
            f"No ONNX model found. Looked for: {[str(p) for p in paths]}. "
            "Run 'archivist export-onnx' to generate the ONNX models."
        )

    @staticmethod
    def _l2_normalize(arr: np.ndarray) -> np.ndarray:
        """L2-normalise rows of a float32 array."""
        norms = np.linalg.norm(arr, axis=-1, keepdims=True).clip(min=1e-8)
        return (arr / norms).astype(np.float32)

    def __repr__(self) -> str:
        loaded = self._vision_session is not None
        return f"ONNXEmbedder(onnx_dir={self.onnx_dir!r}, loaded={loaded})"
