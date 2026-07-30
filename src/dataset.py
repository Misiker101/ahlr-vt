"""
dataset.py -- AHLD-29K loading, vocab construction, and the PyTorch Dataset
used for training/evaluating AHLR-VT.

CHANGED from the original notebook: images are no longer read from
Dataset/train_images/... on disk. They are streamed from your published
Hugging Face dataset (misiker/AHLD-29k) via `datasets.load_dataset`, which
already returns the writer-independent train/validation/test split:

    DatasetDict({
        train:      23911 rows,
        validation:  3545 rows,
        test:        2491 rows,
    })
    features: ['image', 'text', 'file_name', 'writer_id']
"""

import os
import json

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import albumentations as A
from datasets import load_dataset, DatasetDict

HF_DATASET_ID = "misiker/AHLD-29k"

# CHANGED: [BLANK] must stay at index 0 -- CTCLoss(blank=0) and every decode
# function in evaluate.py assume this.
SPECIAL_TOKENS = ["[BLANK]", "<SPACE>", "<UNK>"]


# ---------------------------------------------------------------------------
# Loading the Hugging Face dataset
# ---------------------------------------------------------------------------
def load_ahld29k(cache_dir=None) -> DatasetDict:
    """Downloads/streams misiker/AHLD-29k and returns the DatasetDict with
    train/validation/test splits, exactly as printed in your snippet."""
    ds = load_dataset(HF_DATASET_ID, cache_dir=cache_dir)
    return ds


# ---------------------------------------------------------------------------
# Vocab
# ---------------------------------------------------------------------------
def build_or_load_vocab(hf_dataset_dict: DatasetDict, vocab_path: str = "vocab.json",
                         split_for_vocab: str = "train", force_rebuild: bool = False):
    """
    The original notebook shipped a fixed vocab.txt (313 tokens: 3 special +
    310 Amharic/Latin/punctuation characters seen in the corpus). Since the
    HF dataset doesn't include that file, we derive the vocab once from the
    training split's text column and cache it to `vocab_path` (JSON, ordered
    list) so every subsequent run -- and every teammate cloning the repo --
    gets an IDENTICAL vocab -> identical class indices -> checkpoints stay
    compatible across runs.

    IMPORTANT: commit the generated vocab.json to your GitHub repo after the
    first run, so training/evaluation always uses the same vocab even if the
    HF dataset is later updated/reshuffled.
    """
    if os.path.exists(vocab_path) and not force_rebuild:
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        print(f"Loaded existing vocab ({len(vocab)} tokens) from {vocab_path}")
        return vocab

    charset = set()
    for text in hf_dataset_dict[split_for_vocab]["text"]:
        charset.update(ch for ch in text if ch != " ")

    vocab = SPECIAL_TOKENS + sorted(charset)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Built new vocab with {len(vocab)} tokens from '{split_for_vocab}' split -> saved to {vocab_path}")
    return vocab


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapping one HF split
# ---------------------------------------------------------------------------
class AmharicHFDataset(Dataset):
    """Wraps one split (train/validation/test) of the HF DatasetDict."""

    def __init__(self, hf_split, vocab, img_height=64, augment=False):
        self.ds = hf_split
        self.img_height = img_height
        self.augment = augment

        self.vocab = vocab
        self.char_to_idx = {c: i for i, c in enumerate(vocab)}
        self.idx_to_char = {i: c for i, c in enumerate(vocab)}

        assert self.vocab[0] == "[BLANK]", "vocab[0] must be [BLANK] to match CTCLoss(blank=0)"
        assert "<SPACE>" in self.char_to_idx and "<UNK>" in self.char_to_idx

        if self.augment:
            self.transform = A.Compose([
                A.SafeRotate(limit=2, border_mode=cv2.BORDER_REPLICATE, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.03, scale_limit=(-0.1, 0.0),
                                    rotate_limit=0, border_mode=cv2.BORDER_REPLICATE, p=0.3),
                A.ElasticTransform(alpha=1, sigma=20, border_mode=cv2.BORDER_REPLICATE, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.GaussNoise(p=0.2),
                A.Blur(blur_limit=3, p=0.1),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]

        # CHANGED: image now comes from the HF `Image` feature (decoded to a
        # PIL.Image automatically) instead of cv2.imread() from disk.
        pil_img = sample["image"].convert("L")  # ensure single-channel grayscale
        img = np.array(pil_img)

        text = sample["text"]
        filename = sample.get("file_name") or f"row_{idx}"

        h, w = img.shape
        new_w = max(1, int(w * (self.img_height / h)))
        img = cv2.resize(img, (new_w, self.img_height))

        if self.augment and self.transform is not None:
            img = self.transform(image=img)["image"]

        img = img.astype("float32") / 255.0
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]

        target = torch.tensor(
            [self.char_to_idx.get(ch, self.char_to_idx["<UNK>"])
             if ch != " " else self.char_to_idx["<SPACE>"]
             for ch in text],
            dtype=torch.long,
        )

        return img_tensor, target, text, filename


def collate_fn(batch):
    images, targets, texts, filenames = zip(*batch)
    max_w = max(img.shape[2] for img in images)
    padded_images = []
    for img in images:
        pad_width = max_w - img.shape[2]
        padded_img = torch.nn.functional.pad(img, (0, pad_width, 0, 0), value=1.0)
        padded_images.append(padded_img)
    padded_images = torch.stack(padded_images)

    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)

    return padded_images, padded_targets, target_lengths, texts, filenames


def get_datasets(img_height=64, vocab_path="vocab.json", cache_dir=None):
    """Convenience one-liner used by train.py / evaluate.py / stats.py.

    Returns (train_dataset, val_dataset, test_dataset, vocab).
    """
    hf_ds = load_ahld29k(cache_dir=cache_dir)
    vocab = build_or_load_vocab(hf_ds, vocab_path=vocab_path, split_for_vocab="train")

    train_dataset = AmharicHFDataset(hf_ds["train"], vocab, img_height=img_height, augment=True)
    val_dataset = AmharicHFDataset(hf_ds["validation"], vocab, img_height=img_height, augment=False)
    test_dataset = AmharicHFDataset(hf_ds["test"], vocab, img_height=img_height, augment=False)
    return train_dataset, val_dataset, test_dataset, vocab
