from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download one GGUF model file from Hugging Face into models/.")
    parser.add_argument("--repo", required=True, help="Hugging Face repo, e.g. owner/model-GGUF")
    parser.add_argument("--file", required=True, help="Exact GGUF filename in that repo")
    parser.add_argument("--output", default="models/model.gguf", help="Destination path")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("Install requirements-local-model.txt first") from exc

    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cached = Path(hf_hub_download(repo_id=args.repo, filename=args.file))
    shutil.copy2(cached, dst)
    print(f"Model ready: {dst.resolve()}")


if __name__ == "__main__":
    main()
