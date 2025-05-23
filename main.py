import os
import json
import time
import torch
from config import Config
from data_helper import create_data_loaders
from model import BERTClassifier
from train import Trainer
import warnings
import argparse

warnings.filterwarnings("ignore", category=UserWarning, module="torch.autograd.graph")


def train_classifier(config, labels, classifier_name, log_file):
    start_time = time.time()

    if os.path.exists(log_file):
        os.remove(log_file)

    if classifier_name == "root":
        use_other = False
    else:
        use_other = True
    train_loader, val_loader, test_loader, _ = create_data_loaders(
        config, classifier_name, labels, use_other, log_file
    )

    dataset_info = {
        "train_iterations": len(train_loader),
        "val_iterations": len(val_loader),
        "test_iterations": len(test_loader),
    }

    print(f"\nTraining classifier: {classifier_name}")
    print(f"Train iterations/epoch\t", dataset_info["train_iterations"])
    print(f"Val iterations/epoch\t", dataset_info["val_iterations"])
    print(f"Test iterations/epoch\t", dataset_info["test_iterations"])

    model = BERTClassifier(config).to(config.device)
    model_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {model_params}")

    with open(log_file, "a") as f:
        f.write(f"\n=== Classifier: {classifier_name} ===\n")
        f.write(f"Train iterations/epoch\t {dataset_info['train_iterations']}\n")
        f.write(f"Val iterations/epoch\t {dataset_info['val_iterations']}\n")
        f.write(f"Test iterations/epoch\t {dataset_info['test_iterations']}\n")
        f.write(f"Model parameters: {model_params}\n")

    model = torch.compile(model)
    trainer = Trainer(
        model, config, log_file, classifier_name, train_loader, val_loader
    )
    trainer.train()

    end_time = time.time()
    print(f"\nTotal time: {end_time - start_time:.2f}s")

    with open(log_file, "a") as f:
        f.write(f"Total time: {end_time - start_time:.2f}s\n")


def main():
    start_time = time.time()

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
        torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(
        description="Train classifiers with specified dataset, augmentation flag and checkpoint name"
    )
    parser.add_argument(
        "--ds", dest="dataset", type=str, help="dataset folder name", default=None
    )
    parser.add_argument(
        "--aug", dest="aug", action="store_true", help="enable data augmentation"
    )
    parser.add_argument(
        "--no-aug", dest="aug", action="store_false", help="disable data augmentation"
    )
    parser.set_defaults(aug=True)
    parser.add_argument(
        "--ckpt",
        dest="ckpt",
        type=str,
        help="checkpoint subdirectory name",
        default=None,
    )
    args = parser.parse_args()

    config = Config()
    if args.dataset:
        config.dataset_name = args.dataset
    if args.ckpt:
        config.checkpoint_subdir = args.ckpt
    config.enable_augmentation = args.aug
    config.checkpoint_dir = (
        f"checkpoints/{config.dataset_name}/{config.checkpoint_subdir}"
    )

    log_dir = f"train_logs/{config.dataset_name}/{config.checkpoint_subdir}"
    os.makedirs(log_dir, exist_ok=True)

    with open(f"{config.data_path}/{config.dataset_name}/labels.json", "r") as f:
        all_labels = json.load(f)

    if config.enable_augmentation:
        with open(f"{config.data_path}/{config.dataset_name}/da_labels.json", "r") as f:
            da_labels_json = json.load(f)
        da_labels = []
        for key in da_labels_json.keys():
            da_labels.append(key)
    else:
        da_labels = []

    for key, labels in all_labels.items():
        if config.enable_augmentation:
            if key not in da_labels:
                continue

        key = key.replace("/", "_")
        print(f"\n\n=== Training classifier: {key} ===")
        train_classifier(config, labels, key, os.path.join(log_dir, f"{key}_log.txt"))

    end_time = time.time()
    print(f"\nTotal time: {end_time - start_time:.2f}s")


if __name__ == "__main__":
    main()
