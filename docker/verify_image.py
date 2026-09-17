#!/usr/bin/env python3
"""Verify the packaged runtime without loading model weights or using a GPU."""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path


ROOT = Path("/opt/dsv41")
SGLANG_ROOT = Path("/sgl-workspace/sglang")


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    lock = json.loads((ROOT / "versions.json").read_text())
    target = os.environ.get("DSV41_BUILD_TARGET", "runtime")
    if target not in ("hybrid", "runtime"):
        raise SystemExit(f"Unexpected build target: {target}")
    expected = {
        "sglang": lock["sglang"]["version"],
        "torch": lock["torch"],
        "triton": lock["triton"],
        "tilelang": lock["tilelang"],
        "apache-tvm-ffi": lock["apache_tvm_ffi"],
        "flashinfer-python": lock["flashinfer"]["version"],
        "flashinfer-cubin": lock["flashinfer"]["version"],
        "flashinfer-jit-cache": lock["flashinfer_jit_cache"],
    }
    packages = {name: importlib.metadata.version(name) for name in expected}
    problems = [
        f"{name}: expected {version}, found {packages[name]}"
        for name, version in expected.items()
        if packages[name] != version
    ]
    manifest = json.loads((ROOT / "sglang-source-manifest.json").read_text())
    if manifest["commit"] != lock["sglang"]["commit"]:
        problems.append("SGLang source manifest and version lock disagree")
    for relative, expected_hash in manifest["files"].items():
        path = SGLANG_ROOT / relative
        if not path.is_file() or digest(path) != expected_hash:
            problems.append(f"SGLang source missing or modified: {relative}")
    runtime = json.loads((ROOT / "runtime/provenance.json").read_text())
    for item in runtime:
        path = ROOT / item["file"]
        if not path.is_file() or digest(path) != item["packaged_sha256"]:
            problems.append(f"Runtime extension missing or modified: {item['file']}")

    fi = lock["flashinfer"]
    fi_spec = importlib.util.find_spec("flashinfer")
    fi_root = Path(fi_spec.origin).parent
    source = fi_root / "data/csrc/sparse_mla_sm120_prefill.cu"
    expected_source = (
        fi["source_after_sha256"] if target == "runtime" else fi["source_before_sha256"]
    )
    if digest(source) != expected_source:
        problems.append("FlashInfer CUDA source does not match the selected target")
    backport_manifest = Path("/opt/flashinfer-pr5121/manifest.json")
    backport = None
    if target == "runtime":
        backport = json.loads(backport_manifest.read_text())
        if (
            backport["commit"] != fi["commit"]
            or backport["jit_module"] != fi["jit_module"]
        ):
            problems.append("FlashInfer backport provenance mismatch")
        library = Path(backport["library"])
        if not library.is_file() or digest(library) != backport["library_sha256"]:
            problems.append("Compiled native-prefill library is missing or modified")
        os.environ["FLASHINFER_CUDA_ARCH_LIST"] = fi["cuda_arch"]
        from flashinfer.jit.mla import gen_sparse_mla_sm120_module

        module = gen_sparse_mla_sm120_module()
        if module.name != fi["jit_module"] or module.is_aot:
            problems.append("FlashInfer selected the wrong/stale sparse-MLA module")
        if module.get_library_path().resolve() != library.resolve():
            problems.append("FlashInfer library lookup differs from the build manifest")
    elif backport_manifest.exists():
        problems.append("Hybrid target unexpectedly contains the native backport")

    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    from sglang.srt.parser.reasoning_parser import ReasoningParser

    tool_parser = FunctionCallParser([], "deepseekv41")
    reasoning_parser = ReasoningParser("deepseek-v41")
    if problems:
        raise SystemExit("\n".join(problems))
    print(
        json.dumps(
            {
                "status": "passed",
                "target": target,
                "gpu_used": False,
                "model_loaded": False,
                "sglang_commit": manifest["commit"],
                "source_files_verified": len(manifest["files"]),
                "runtime_files_verified": len(runtime),
                "packages": packages,
                "tool_detector": type(tool_parser.detector).__name__,
                "reasoning_detector": type(reasoning_parser.detector).__name__,
                "backport": backport,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
