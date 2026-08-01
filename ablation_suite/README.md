## Ablation & Depth Comparison Suite

In addition to our modular pipeline in `src/`, we provide a standalone, consolidated script located in `ablation_suite/ahld_ablation_pipeline.py`. This script is designed for end-to-end reproducibility of the stem-ablation and depth-ablation studies presented in our research.

we can train all the models (including the proposed) only using this file without the main module `src` along with different variants. Use this especially to experiment different components like pure vit or vit depth it is convenient.

It unifies **both** architecture families across 5 transformer encoder depths ($d \in \{2, 4, 6, 8, 12\}$):
1. **AHLR-VT (Hybrid CNN + ViT):** Uses our 4-block asymmetric CNN feature extractor combined with a truncated ViT encoder.
2. **Pure-ViT (No CNN):** Replaces the CNN stem with a single non-overlapping 1D patch embedding ($H \times 4$), maintaining exact token sequence lengths for a direct structural comparison.

### Key Features
* **Local Data Pipeline:** Loads directly from local image directories (`dataset/`) and local vocabulary files (`vocab.txt`) using OpenCV for consistent pixel decoding.
* **Complete Evaluation Suite:** Evaluates test loss, Corpus CER/WER, 95% Bootstrap Confidence Intervals, paired statistical significance testing (t-test & Wilcoxon signed-rank test), character substitution matrix, and CTC beam vs. greedy search decoding.

---

### Usage Instructions

#### 1. Setup Local Dataset & Vocab
Ensure your local dataset follows the expected directory structure:
```text
dataset/
├── train_images/output_lines/
├── val_images/output_linesval/
└── test_images/output_linestest_cleaned/
vocab.txt
```

#### Usage
change directory to `cd ablation_suite`

Train the proposed model:

    
    python ahlr_vt_pipeline.py train --variant AHLR-VT --fresh
    
    or if using without changing the dir

    python ablation_suite/ahlr_vt_pipeline.py train --variant AHLR-VT

Train the Hybrid-CNN-ViT depth-ablation variants:

    
    python ahlr_vt_pipeline.py train --variant Hybrid-ViT-d8
    python ahlr_vt_pipeline.py train --variant Hybrid-ViT-d6
    python ahlr_vt_pipeline.py train --variant Hybrid-ViT-d4
    python ahlr_vt_pipeline.py train --variant Hybrid-ViT-d2
    

Train the Pure-ViT (no CNN) stem-ablation variants:

    
    python ahlr_vt_pipeline.py train --variant Pure-ViT-d12
    python ahlr_vt_pipeline.py train --variant Pure-ViT-d8
    python ahlr_vt_pipeline.py train --variant Pure-ViT-d6
    python ahlr_vt_pipeline.py train --variant Pure-ViT-d4
    python ahlr_vt_pipeline.py train --variant Pure-ViT-d2
    

Run the full Part-2 suite across every variant trained so far:

    python ahlr_vt_pipeline.py validate

    or

    python ablation_suite/ahlr_vt_pipeline.py validate