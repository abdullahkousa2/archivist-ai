"""
ONNX export utility for Archivist-AI embedding models.

Exports the SigLIP (or CLIP) vision and text encoders to ONNX format,
then applies dynamic int8 quantization using onnxruntime-tools for
significantly faster CPU inference (3-5x vs PyTorch).

Typical workflow
----------------
    # One-time export (takes ~60s)
    archivist export-onnx

    # Subsequent runs use the ONNX engine automatically when use_onnx=true
    archivist search "pizza"

Output files
------------
    ~/.archivist/onnx/
        vision_encoder.onnx           # full-precision vision encoder
        vision_encoder_quant.onnx     # int8-quantized vision encoder (used at runtime)
        text_encoder.onnx             # full-precision text encoder
        text_encoder_quant.onnx       # int8-quantized text encoder (used at runtime)
        processor_id.txt              # model id for processor loading
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Default ONNX output directory
DEFAULT_ONNX_DIR = Path.home() / ".archivist" / "onnx"


# ── Wrapper modules ───────────────────────────────────────────────────────────


def _make_vision_wrapper(model, is_siglip: bool):
    """Thin nn.Module that returns only the pooled vision embedding."""
    import torch.nn as nn

    if is_siglip:
        class _VisionEncoder(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, pixel_values):
                # Use vision_model directly — get_image_features() returns
                # BaseModelOutputWithPooling in newer transformers, which
                # would create multiple ONNX outputs.
                out = self.m.vision_model(pixel_values=pixel_values, return_dict=True)
                return out.pooler_output  # (N, 768)
    else:
        class _VisionEncoder(nn.Module):  # type: ignore[no-redef]
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, pixel_values):
                out = self.m.vision_model(pixel_values=pixel_values, return_dict=True)
                pooled = out.pooler_output
                return self.m.visual_projection(pooled)  # (N, 512)

    return _VisionEncoder(model).eval()


def _make_text_wrapper(model, is_siglip: bool):
    """Thin nn.Module that returns only the pooled text embedding."""
    import torch.nn as nn

    if is_siglip:
        class _TextEncoder(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, input_ids, attention_mask):
                out = self.m.text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                return out.pooler_output  # (N, 768)
    else:
        class _TextEncoder(nn.Module):  # type: ignore[no-redef]
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, input_ids, attention_mask):
                out = self.m.text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                pooled = out.pooler_output
                return self.m.text_projection(pooled)  # (N, 512)

    return _TextEncoder(model).eval()


# ── Core export function ──────────────────────────────────────────────────────


def export_to_onnx(
    model_id: str,
    output_dir: Path | str = DEFAULT_ONNX_DIR,
    opset: int = 14,
    quantize: bool = True,
) -> Path:
    """
    Load the HuggingFace model and export vision + text encoders to ONNX.

    Parameters
    ----------
    model_id:   HuggingFace model id (e.g. "google/siglip-base-patch16-224")
    output_dir: Where to write the .onnx files
    opset:      ONNX opset version (14 is a safe default for ort >= 1.16)
    quantize:   Apply onnxruntime int8 dynamic quantization after export

    Returns
    -------
    output_dir Path (all files written there)
    """
    import torch
    from transformers import AutoProcessor

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    is_siglip = "siglip" in model_id.lower()

    logger.info(f"Loading {model_id} for ONNX export…")

    # Load model WITHOUT quantization (ONNX export requires float32 weights)
    if is_siglip:
        from transformers import SiglipModel, SiglipProcessor
        processor = SiglipProcessor.from_pretrained(model_id)
        model = SiglipModel.from_pretrained(model_id).eval()
    else:
        from transformers import CLIPModel, CLIPProcessor
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id).eval()

    # ── Vision encoder ────────────────────────────────────────────────────────
    vision_path = output_dir / "vision_encoder.onnx"
    logger.info("Exporting vision encoder…")

    if is_siglip:
        # SigLIP uses 224×224 images
        dummy_pixels = torch.zeros(1, 3, 224, 224)
    else:
        # CLIP ViT-B/32 also uses 224×224
        dummy_pixels = torch.zeros(1, 3, 224, 224)

    vision_wrapper = _make_vision_wrapper(model, is_siglip)

    import torch
    with torch.no_grad():
        torch.onnx.export(
            vision_wrapper,
            (dummy_pixels,),
            str(vision_path),
            input_names=["pixel_values"],
            output_names=["image_features"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "image_features": {0: "batch_size"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )
    logger.info(f"  ✓ vision_encoder.onnx  ({vision_path.stat().st_size // 1024 // 1024} MB)")

    # ── Text encoder ──────────────────────────────────────────────────────────
    text_path = output_dir / "text_encoder.onnx"
    logger.info("Exporting text encoder…")

    # Determine sequence length from processor
    if is_siglip:
        seq_len = 64   # SigLIP default max_length
    else:
        seq_len = 77   # CLIP default max_length

    dummy_ids = torch.zeros(1, seq_len, dtype=torch.long)
    dummy_mask = torch.ones(1, seq_len, dtype=torch.long)
    text_wrapper = _make_text_wrapper(model, is_siglip)

    with torch.no_grad():
        torch.onnx.export(
            text_wrapper,
            (dummy_ids, dummy_mask),
            str(text_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["text_features"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "attention_mask": {0: "batch_size", 1: "seq_len"},
                "text_features": {0: "batch_size"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )
    logger.info(f"  ✓ text_encoder.onnx  ({text_path.stat().st_size // 1024 // 1024} MB)")

    # ── Save processor id ─────────────────────────────────────────────────────
    (output_dir / "processor_id.txt").write_text(model_id)

    # ── Optional: quantize with onnxruntime ───────────────────────────────────
    if quantize:
        _quantize_onnx(vision_path, output_dir / "vision_encoder_quant.onnx")
        _quantize_onnx(text_path, output_dir / "text_encoder_quant.onnx")

    logger.info(f"ONNX export complete → {output_dir}")
    return output_dir


def _quantize_onnx(input_path: Path, output_path: Path) -> None:
    """Apply onnxruntime dynamic int8 quantization to an ONNX model."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(
            str(input_path),
            str(output_path),
            weight_type=QuantType.QInt8,
        )
        size_mb = output_path.stat().st_size // 1024 // 1024
        logger.info(f"  ✓ {output_path.name}  ({size_mb} MB, int8)")
    except ImportError:
        logger.warning(
            "onnxruntime-tools not installed — skipping quantization. "
            "Install with: pip install onnxruntime"
        )
    except Exception as exc:
        logger.warning(f"ONNX quantization failed ({exc}); non-quantized model will be used.")


def onnx_models_exist(onnx_dir: Path) -> bool:
    """Return True if exported ONNX models are present in onnx_dir."""
    onnx_dir = Path(onnx_dir)
    # Accept either quantized or non-quantized vision+text pairs
    has_vision = (
        (onnx_dir / "vision_encoder_quant.onnx").exists()
        or (onnx_dir / "vision_encoder.onnx").exists()
    )
    has_text = (
        (onnx_dir / "text_encoder_quant.onnx").exists()
        or (onnx_dir / "text_encoder.onnx").exists()
    )
    return has_vision and has_text
