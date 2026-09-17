"""Auto-install the SM120 patch in every interpreter that starts with this dir on PYTHONPATH.

SGLang forks worker processes that would each miss a patch applied only in the
parent, so the hook has to run at interpreter start rather than from a launcher.
Failures are swallowed: a broken patch must not stop the server from booting --
it falls back to SGLang's own SM120 path.

Disable attention with DSV41_SM120_DISABLE=1. The tuned FP8 GEMM is separate:
DSV41_SM120_FP8_DISABLE=0 enables it; older launch snapshots leave it disabled.
"""

import os
import sys

if os.environ.get("DSV41_SM120_DISABLE", "0") not in ("1", "true", "yes"):
    try:
        import sglang_sm120_patch

        sglang_sm120_patch.install()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[sglang_sm120_patch/sitecustomize] skipped: {exc}\n")

if os.environ.get("DSV41_SM120_FP8_DISABLE", "1") != "1":
    try:
        import sm120_fp8_patch

        sm120_fp8_patch.install()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[sm120_fp8_patch/sitecustomize] skipped: {exc}\n")
