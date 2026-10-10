"""Write non-secret runtime metadata used as reproducibility evidence."""

from __future__ import annotations

import json

from src.config import REPO_ROOT, runtime_metadata


def main() -> None:
    """Write current runtime details to a tracked JSON evidence file."""
    output_path = REPO_ROOT / "docs" / "runtime_metadata.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(runtime_metadata(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote runtime metadata to: {output_path}")


if __name__ == "__main__":
    main()