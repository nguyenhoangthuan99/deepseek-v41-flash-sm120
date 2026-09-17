"""Apply the minimal FlashInfer #5121 backport and isolate its JIT cache."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import flashinfer

pins = json.loads(Path("/opt/dsv41/versions.json").read_text())["flashinfer"]
root = Path(flashinfer.__file__).resolve().parent
source = root / "data/csrc/sparse_mla_sm120_prefill.cu"
version = importlib.metadata.version("flashinfer-python")
if version != pins["version"]:
    raise RuntimeError(f"FlashInfer version mismatch: {version}")
before = hashlib.sha256(source.read_bytes()).hexdigest()
if before != pins["source_before_sha256"]:
    raise RuntimeError(f"Unrecognized FlashInfer source: {before}")
patch = "/opt/flashinfer-pr5121/pr5121-0.6.18.patch"
subprocess.run(["git", "apply", "--check", patch], cwd=root / "data", check=True)
subprocess.run(["git", "apply", patch], cwd=root / "data", check=True)
after = hashlib.sha256(source.read_bytes()).hexdigest()
if after != pins["source_after_sha256"]:
    raise RuntimeError(f"Patched FlashInfer source mismatch: {after}")

# Source changes alone cannot bypass the installed prebuilt JIT-cache wheel.
# Rename only this artifact; leave all other kernels and the public API intact.
jit_file = root / "jit/mla.py"
jit_source = jit_file.read_text()
old_name = '"sparse_mla_sm120"'
if jit_source.count(old_name) != 1:
    raise RuntimeError("Expected exactly one sparse-MLA JIT module declaration")
jit_file.write_text(jit_source.replace(old_name, json.dumps(pins["jit_module"]), 1))
manifest = {
    "pr": "https://github.com/flashinfer-ai/flashinfer/pull/5121",
    "commit": pins["commit"],
    "backport": "Only the PR's 128/256 extra-page full-tile and length-aware dispatch additions, adapted to the 0.6.18 template signatures",
    "flashinfer_version": version,
    "source": str(source),
    "source_before_sha256": before,
    "source_after_sha256": after,
    "jit_module": pins["jit_module"],
    "cache_isolation": "Only the sparse-MLA JIT artifact name changes; no dependency upgrade",
}
Path("/opt/flashinfer-pr5121/manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)
print(json.dumps(manifest), flush=True)
