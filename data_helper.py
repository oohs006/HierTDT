import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import re
from pathlib import Path


def tokenize_label(label_list, tokenizer):
    cls_id = torch.tensor([[101]])
    sep_id = torch.tensor([[102]])
    unmask = torch.ones((1, 1))

    label_input_ids = []
    label_attention_mask = []
    label_token_type_ids = []

    for detail in label_list:
        label_tokens = tokenizer(
            detail,
            add_special_tokens=False,
            max_length=3,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        label_input_ids.append(label_tokens["input_ids"])
        label_attention_mask.append(label_tokens["attention_mask"])
        label_token_type_ids.append(torch.zeros_like(label_tokens["input_ids"]))

    label_input_ids = torch.cat(label_input_ids, dim=-1)
    label_input_ids = torch.cat([cls_id, label_input_ids, sep_id], dim=-1)

    label_attention_mask = torch.cat(label_attention_mask, dim=-1)
    label_attention_mask = torch.cat([unmask, label_attention_mask, unmask], dim=-1)

    label_token_type_ids = torch.cat(label_token_type_ids, dim=-1)
    label_token_type_ids = torch.cat(
        [torch.zeros_like(cls_id), label_token_type_ids, torch.zeros_like(sep_id)],
        dim=-1,
    )

    return {
        "input_ids": label_input_ids,
        "attention_mask": label_attention_mask,
        "token_type_ids": label_token_type_ids,
    }


class CustomDataset(Dataset):
    def __init__(self, texts, labels, unique_labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_tokens = tokenize_label(unique_labels, tokenizer)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        text = clean_text(text)

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "text_input_ids": encoding["input_ids"].flatten(),
            "text_attention_mask": encoding["attention_mask"].flatten(),
            "text_token_type_ids": encoding["token_type_ids"].flatten(),
            "label_input_ids": self.label_tokens["input_ids"].flatten(),
            "label_attention_mask": self.label_tokens["attention_mask"].flatten(),
            "label_token_type_ids": self.label_tokens["token_type_ids"].flatten(),
            "labels": torch.FloatTensor(self.labels[idx]),
        }


def clean_text(text):
    text = text.strip().strip('"')
    text = text.replace('\\"', '"')

    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http[s]?://\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r'[^a-z0-9\s.,!?\'"()\$-]', " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_dataset(config, parent_label, subtree_labels, use_other=True, log_file=None):
    base = Path(config.data_path) / config.dataset_name

    def load_file(filename):
        texts, labels = [], []
        for line in (base / filename).open(encoding="utf-8"):
            item = json.loads(line)
            texts.append(item["text"])
            labels.append(item["label"])
        return texts, labels

    raw = {}
    ds = config.dataset_name.lower()
    if ds == "rcv1":
        for split in ("train", "val", "test"):
            fname = (
                f"rcv1_sampled_{split}.jsonl"
                if split == "test"
                else f"rcv1_{split}.jsonl"
            )
            raw[split] = load_file(fname)
    elif ds == "wos":
        for split in ("train", "val", "test"):
            raw[split] = load_file(f"wos_{split}.jsonl")
    elif ds == "nyt":
        for split in ("train", "val", "test"):
            raw[split] = load_file(f"nyt_{split}.jsonl")
    else:
        raise ValueError(f"Unknown dataset: {config.dataset_name}")
    da_texts, da_labels = [], []
    if config.enable_augmentation:
        for line in (base / "da_data.jsonl").open(encoding="utf-8"):
            item = json.loads(line)
            if parent_label in item["da_label"]:
                da_texts.append(item["text"])
                da_labels.append(item["da_label"])
    if use_other:
        all_labels = sorted(subtree_labels) + ["other"]
    else:
        all_labels = sorted(subtree_labels)
    label2id = {l: i for i, l in enumerate(all_labels)}
    other_id = label2id.get("other")
    n = len(all_labels)
    subtree_set = set(subtree_labels)

    def encode(label_sets):
        out = []
        for lbls in label_sets:
            mh = [0] * n
            for l in lbls:
                if l in subtree_set:
                    mh[label2id[l]] = 1
            if use_other and not any(mh):
                mh[other_id] = 1
            out.append(mh)
        return out

    dfs = {}
    for split, (texts, labels) in raw.items():
        dfs[split] = pd.DataFrame(
            {
                "text": texts,
                "original_labels": labels,
                "labels": encode(labels),
            }
        )
    if da_texts:
        da_df = pd.DataFrame(
            {
                "text": da_texts,
                "original_labels": da_labels,
                "labels": encode(da_labels),
            }
        )
        dfs["train"] = (
            pd.concat([dfs["train"], da_df], ignore_index=True)
            .sample(frac=1)
            .reset_index(drop=True)
        )
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"labels: {len(all_labels)}\n")
            f.write(f"da: {len(da_texts)}\n")
            f.write(f"train: {len(dfs['train'])}\n")
            f.write(f"val: {len(dfs['val'])}\n")
            f.write(f"test: {len(dfs['test'])}\n")
    config.num_labels = len(all_labels)
    id2label = {i: l for l, i in label2id.items()}
    return dfs["train"], dfs["val"], dfs["test"], all_labels, id2label


def create_data_loaders(
    config, parent_label, subtree_labels, use_other=True, log_file=None
):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    train_df, val_df, test_df, all_labels, id2label = split_dataset(
        config, parent_label, subtree_labels, use_other, log_file
    )
    train_dataset = CustomDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["labels"].tolist(),
        unique_labels=all_labels,
        tokenizer=tokenizer,
        max_length=config.max_length,
    )
    val_dataset = CustomDataset(
        texts=val_df["text"].tolist(),
        labels=val_df["labels"].tolist(),
        unique_labels=all_labels,
        tokenizer=tokenizer,
        max_length=config.max_length,
    )
    test_dataset = CustomDataset(
        texts=test_df["text"].tolist(),
        labels=test_df["labels"].tolist(),
        unique_labels=all_labels,
        tokenizer=tokenizer,
        max_length=config.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, id2label
