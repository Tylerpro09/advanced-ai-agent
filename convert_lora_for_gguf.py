from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a PEFT LoRA adapter to llama.cpp GGUF format.")
    parser.add_argument("--adapter", required=True, help="PEFT adapter directory created by train_lora.py")
    parser.add_argument("--base", required=True, help="Original Hugging Face base model directory/repo materialized locally")
    parser.add_argument("--out", required=True, help="Output .gguf LoRA adapter path")
    parser.add_argument("--llama-cpp-dir", default=os.getenv("LLAMA_CPP_DIR", ""), help="llama.cpp repository containing convert_lora_to_gguf.py")
    args = parser.parse_args()

    root = Path(args.llama_cpp_dir).expanduser().resolve() if args.llama_cpp_dir else None
    candidates = []
    if root:
        candidates += [root / "convert_lora_to_gguf.py", root / "convert_lora_to_gguf.py"]
    script = next((x for x in candidates if x.is_file()), None)
    if script is None:
        raise SystemExit("Could not find convert_lora_to_gguf.py. Clone llama.cpp and pass --llama-cpp-dir or set LLAMA_CPP_DIR.")

    adapter = Path(args.adapter).expanduser().resolve()
    base = Path(args.base).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not adapter.is_dir():
        raise SystemExit(f"Adapter directory not found: {adapter}")
    if not base.exists():
        raise SystemExit(f"Base model path not found: {base}")
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(script), str(adapter), "--base", str(base), "--outfile", str(out)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Converted LoRA:", out)
    print("Set LOCAL_LORA_PATH=" + str(out))


if __name__ == "__main__":
    main()
