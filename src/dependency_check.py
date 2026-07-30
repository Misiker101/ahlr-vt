import sys


def check_dependencies():
    try:
        import torch                
        import torchvision          
        import cv2                  
        import albumentations       
        import Levenshtein          
        import pandas               
        import numpy                
        import scipy                
        import datasets             
        import PIL                  
    except ImportError as e:
        print(f"\n[Error] Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt\n")
        sys.exit(1)
