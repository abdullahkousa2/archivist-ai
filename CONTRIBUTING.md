# Contributing to Archivist-AI

Thank you for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/yourusername/archivist-ai
cd archivist-ai
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v
```

## Code style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
ruff check archivist/
ruff format archivist/
```

The CI will fail if linting errors are present.

## Before opening a PR

1. **Open an issue first** for anything larger than a bug fix — so we can discuss the approach before you invest time coding it.
2. Make sure `pytest` passes and `ruff check` is clean.
3. Add or update tests for any new behaviour.
4. Keep PRs focused. One logical change per PR is much easier to review.

## Project structure

```
archivist/
├── core/           # Engine: embedder, store (FAISS+SQLite), indexer, searcher
├── features/       # Higher-level features: duplicates, autotag, organizer, watcher
├── ui/             # Gradio web interface
├── cli.py          # Typer CLI entry point
└── config.py       # ArchivistConfig dataclass + JSON persistence

tests/              # pytest test suite
```

## Reporting bugs

Use the [Bug Report](.github/ISSUE_TEMPLATE/bug_report.md) template. Include your OS, Python version, and the full error traceback.

## Feature requests

Use the [Feature Request](.github/ISSUE_TEMPLATE/feature_request.md) template. Explain the use case — not just the feature.

## License

By contributing you agree that your contributions will be licensed under the MIT License.
