"""
Configuration management for Archivist-AI.

Config is stored at ~/.archivist/config.json and can be overridden
per-command via CLI flags.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_INDEX_DIR = Path.home() / ".archivist" / "index"
DEFAULT_ONNX_DIR = Path.home() / ".archivist" / "onnx"
DEFAULT_MODEL_ID = "google/siglip-base-patch16-224"   # upgraded from CLIP
DEFAULT_EMBEDDING_DIM = 768                             # SigLIP dim (CLIP = 512)
DEFAULT_BATCH_SIZE = 16
DEFAULT_TOP_K = 20
DEFAULT_SIMILARITY_THRESHOLD = 0.0
DEFAULT_DUPLICATE_THRESHOLD = 0.97


@dataclass
class ArchivistConfig:
    """
    Central configuration for all Archivist-AI components.

    Loaded from ~/.archivist/config.json; missing keys fall back to defaults.
    CLI flags always override the loaded config.
    """

    # Storage
    index_dir: Path = field(default_factory=lambda: DEFAULT_INDEX_DIR)

    # Model
    model_id: str = DEFAULT_MODEL_ID
    embedding_dim: int = DEFAULT_EMBEDDING_DIM   # must match the model (768 for SigLIP, 512 for CLIP)
    device: str = "cpu"
    quantize: bool = True          # Apply dynamic int8 quantization for faster CPU

    # ONNX acceleration (3-5x faster than PyTorch on CPU after one-time export)
    use_onnx: bool = False         # Set True after running 'archivist export-onnx'
    onnx_dir: Path = field(default_factory=lambda: DEFAULT_ONNX_DIR)

    # Indexing
    batch_size: int = DEFAULT_BATCH_SIZE
    recursive: bool = True

    # Search
    top_k: int = DEFAULT_TOP_K
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD

    # Duplicate detection
    duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD

    # Auto-tagging
    autotag_on_index: bool = False  # Set True to tag images during indexing (slower)
    autotag_top_k: int = 5

    # ── Serialisation ─────────────────────────────────────────────────────────

    @classmethod
    def load(cls, config_path: Path | None = None) -> "ArchivistConfig":
        """Load config from JSON file, falling back to defaults for missing keys."""
        if config_path is None:
            config_path = Path.home() / ".archivist" / "config.json"

        if not config_path.exists():
            return cls()

        try:
            with open(config_path, encoding="utf-8") as f:
                data: dict = json.load(f)

            # Coerce path strings back to Path objects
            if "index_dir" in data:
                data["index_dir"] = Path(data["index_dir"])
            if "onnx_dir" in data:
                data["onnx_dir"] = Path(data["onnx_dir"])

            # Only pass keys that exist in the dataclass
            valid_keys = cls.__dataclass_fields__.keys()
            filtered = {k: v for k, v in data.items() if k in valid_keys}
            return cls(**filtered)

        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning(f"Could not read config at {config_path}: {exc}. Using defaults.")
            return cls()

    def save(self, config_path: Path | None = None) -> None:
        """Persist config to JSON file."""
        if config_path is None:
            config_path = Path.home() / ".archivist" / "config.json"

        config_path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["index_dir"] = str(data["index_dir"])  # Path → str for JSON

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Config saved to {config_path}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"ArchivistConfig("
            f"index_dir={self.index_dir}, "
            f"model={self.model_id}, "
            f"device={self.device}, "
            f"batch_size={self.batch_size})"
        )
