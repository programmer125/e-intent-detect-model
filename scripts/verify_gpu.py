import json
import platform
import sys

import torch


def main() -> int:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()

    print(json.dumps(info, ensure_ascii=False, indent=2))

    if not torch.cuda.is_available():
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
