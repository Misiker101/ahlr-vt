# AHLR-VT: A Hybrid CNN–Vision Transformer for Offline Amharic Handwritten Text-Line Recognition

Official code for **AHLR-VT**, a hybrid CNN + ViT-Base/16 + CTC architecture for
line-level offline handwritten text recognition (HTR) in Amharic (Ge'ez script),
together with the depth-ablation family used for the parameter-scaled efficiency
comparison in the paper.

- **Paper:** AHLR-VT: A Hybrid CNN–Vision Transformer Architecture for Offline
  Amharic Handwritten Text-Line Recognition
- **Dataset:** [AHLD-29K on Hugging Face](https://huggingface.co/datasets/misiker/AHLD-29k)
  — we created a new benchmark dataset, 29,947 writer-independent, line-level handwritten Amharic text-line images
  from 180 writers, split 80:10:10 (train/validation/test).
- **Author:** Misiker Kassahun Zewde

---

## Table of contents

1. [Repository structure](#1-repository-structure)
2. [Prerequisites](#2-prerequisites)
3. [Step-by-step setup](#3-step-by-step-setup)
4. [Training](#4-training)
5. [Evaluation & statistical validation](#5-evaluation--statistical-validation)
6. [Citation](#6-citation)
7. [License](#7-license)

---

## 1. Repository structure

```
ahlr-vt/
├── README.md                  
├── requirements.txt           
├── LICENSE                    
├── .gitignore                 
├── vocab.json                 
├── src/
│   ├── __init__.py
│   ├── dependency_check.py    
│   ├── dataset.py             <- loads AHLD-29K from Hugging Face, builds vocab, PyTorch Dataset
│   ├── model.py                <- HybridCNNViT architecture + variant registry (depth 2/4/6/8/12)
│   ├── train.py                <- training loop with real early stopping (CLI entry point)
│   ├── evaluate.py             <- CTC metrics/decoding, per-line logged evaluation, efficiency profiling
│   └── stats.py                <- bootstrap CI, paired significance tests, confusion analysis, length robustness, greedy-vs-beam (CLI entry point)        
├── checkpoints/                <- created automatically; holds *.pth files
├── results/                    <- created automatically; holds all output CSVs
└── runs/                       <- created automatically; TensorBoard logs
```


---

## 2. Prerequisites

- Python 3.10.11
- A CUDA-capable GPU is strongly recommended for training

---

## 3. Step-by-step setup

### 3.1 Clone (or create) the project folder

setting it up fresh:

```bash
git clone https://github.com/Misiker101/ahlr-vt.git
cd ahlr-vt
```

### 3.2 Create and activate a virtual environment

```bash
python3 -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

You should see `(.venv)` appear at the start of your terminal prompt once it's active.

### 3.3 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```


---

## 4. Training

Training is **one architecture variant per command**. Run each one variant one at a time


### 4.1 Train the model (proposed model is AHLR-VT) / parameter-scaled variants

```bash
python -m src.train --variant Hybrid-ViT-d8
python -m src.train --variant Hybrid-ViT-d6
python -m src.train --variant Hybrid-ViT-d4
python -m src.train --variant Hybrid-ViT-d2
python -m src.train --variant AHLR-VT
```

Run these sequentially, in whatever order you like just make sure
`AHLR-VT` gets trained too, since the significance tests, confusion
analysis, length-robustness study, and beam-search comparison in Section 5
are all defined **relative to** AHLR-VT and get skipped if it isn't trained.


### 4.2 Useful training flags

```bash
python -m src.train --variant AHLR-VT \
    --max-epochs 100 \
    --patience 10 \
    --batch-size 8 \
    --accumulation-steps 4 \
    --lr 1e-4 \
    --weight-decay 1e-5 \
    --seed 42
```

---

## 5. Evaluation & statistical validation

Once you've trained one or more variants, run:

```bash
python -m src.stats
```

---

## 6. Citation

If you use this code or the AHLD-29K dataset, please cite:

```bibtex
@article{zewde2026ahlrvt,
  title   = {AHLR-VT: A Hybrid CNN-Vision Transformer Architecture for Offline Amharic Handwritten Text-Line Recognition},
  author  = {Zewde, Misiker Kassahun},
  year    = {2026},
}
```

## 7. License

This project is released under the [MIT License](LICENSE).
