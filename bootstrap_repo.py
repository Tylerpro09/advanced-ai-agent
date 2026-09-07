from __future__ import annotations
import base64, io, lzma, tarfile
from pathlib import Path

root = Path(__file__).resolve().parent
parts = sorted(root.glob("bootstrap_payload_*.txt"))
payload = "".join(p.read_text(encoding="ascii").strip() for p in parts)
data = lzma.decompress(base64.b64decode(payload))
with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tf:
    tf.extractall(root)
for p in parts:
    p.unlink(missing_ok=True)
(root / "bootstrap_repo.py").unlink(missing_ok=True)
(root / ".github/workflows/bootstrap.yml").unlink(missing_ok=True)
print("Advanced AI Agent v2.5 extracted successfully")
