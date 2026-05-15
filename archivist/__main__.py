"""Enables `py -3.13 -m archivist` as an alternative to the `archivist` CLI command."""

from archivist.cli import app

if __name__ == "__main__":
    app()
