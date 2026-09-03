"""Verify the GPU/torch pairing before burning a Slurm allocation.

The trap this catches: Blackwell (RTX PRO 6000, GB202) is compute capability
sm_120. Torch builds against CUDA 12.1/12.6 carry no sm_120 kernels, so they
import fine, report cuda.is_available() == True, and then fail at the first
matmul with "no kernel image is available for execution on the device".
"""

import sys


def main():
    try:
        import torch
    except ImportError:
        print("FAIL: torch not installed")
        return 1

    print("torch           :", torch.__version__)
    print("built for CUDA  :", torch.version.cuda)
    print("cuda available  :", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("FAIL: no CUDA device visible")
        return 1

    arches = torch.cuda.get_arch_list()
    print("kernel arches   :", ", ".join(arches))
    n = torch.cuda.device_count()
    print("devices         :", n)

    ok = True
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        tag = "sm_%d%d" % cap
        supported = tag in arches or any(a.startswith("sm_%d" % cap[0]) for a in arches)
        print("  [%d] %-42s %-7s %6.1f GB  %s"
              % (i, name, tag, total, "OK" if supported else "NO KERNELS"))
        if not supported:
            ok = False

    print("bf16 supported  :", torch.cuda.is_bf16_supported())

    if not ok:
        print("\nFAIL: this torch build has no kernels for the installed GPU.")
        print("      Blackwell needs a cu128 build: ")
        print("      pip install torch --index-url https://download.pytorch.org/whl/cu128")
        return 1

    # actually execute something, since arch lists can still lie
    try:
        x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        (x @ x).sum().item()
        torch.cuda.synchronize()
        print("\nsmoke matmul    : OK (bf16)")
    except Exception as exc:  # noqa: BLE001
        print("\nFAIL: smoke matmul failed: %s" % exc)
        return 1
    print("environment looks good.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
