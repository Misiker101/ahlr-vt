"""
dependency_check.py -- run once, first, at the top of every CLI entry point
(train.py, stats.py) BEFORE any heavy imports (torch, torchvision, etc.).
If someone forgets `pip install -r requirements.txt`, this prints a clear
message and exits instead of dying on a confusing ImportError/traceback
somewhere in the middle of the script.
"""

import sys


def check_dependencies():
    try:
        import torch                # noqa: F401
        import torchvision          # noqa: F401
        import cv2                  # noqa: F401
        import albumentations       # noqa: F401
        import Levenshtein          # noqa: F401
        import pandas               # noqa: F401
        import numpy                # noqa: F401
        import scipy                # noqa: F401
        import datasets             # noqa: F401  (Hugging Face `datasets`)
        import PIL                  # noqa: F401  (Pillow, used internally by `datasets` Image feature)
    except ImportError as e:
        print(f"\n[Error] Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt\n")
        sys.exit(1)
