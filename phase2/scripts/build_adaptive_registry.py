"""Rebuild adaptive_registry.json from checkpoint files on disk."""
from adaptive_registry import write_registry

if __name__ == "__main__":
    path = write_registry()
    print(f"Wrote {path}")
