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


SPECIAL_TOKENS = ["[BLANK]", "<SPACE>", "<UNK>"]



# Loading the Hugging Face dataset
def load_ahld29k(cache_dir=None) -> DatasetDict:
    ds = load_dataset(HF_DATASET_ID, cache_dir=cache_dir)
    return ds



# Vocab generation
def build_or_load_vocab(hf_dataset_dict: DatasetDict, vocab_path: str = "vocab.json",
                         split_for_vocab: str = "train", force_rebuild: bool = False):
    
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


# PyTorch Dataset wrapping one HF split
# Wraps one split (train/validation/test) of the HF DatasetDict.
class AmharicHFDataset(Dataset):

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

        
        pil_img = sample["image"].convert("L")  
        img = np.array(pil_img)

        text = sample["text"]
        filename = sample.get("file_name") or f"row_{idx}"

        if self.augment and self.transform is not None:
            img = self.transform(image=img)["image"]

        h, w = img.shape
        new_w = max(1, int(w * (self.img_height / h)))
        img = cv2.resize(img, (new_w, self.img_height))

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

# Convenience one-liner used by train.py / evaluate.py / stats.py.
def get_datasets(img_height=64, vocab_path="vocab.json", cache_dir=None):
    
    hf_ds = load_ahld29k(cache_dir=cache_dir)
    vocab = build_or_load_vocab(hf_ds, vocab_path=vocab_path, split_for_vocab="train")

    train_dataset = AmharicHFDataset(hf_ds["train"], vocab, img_height=img_height, augment=True)
    val_dataset = AmharicHFDataset(hf_ds["validation"], vocab, img_height=img_height, augment=False)
    test_dataset = AmharicHFDataset(hf_ds["test"], vocab, img_height=img_height, augment=False)
    return train_dataset, val_dataset, test_dataset, vocab
