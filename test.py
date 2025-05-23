import json
import os
import time
import torch
from config import Config
from data_helper import create_data_loaders
from model import BERTClassifier
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
config = Config()


@torch.no_grad()
def test(model, parent, id2label, data_loader, threshold=0.5):
    model.eval()

    all_probs = []
    all_labels = []
    all_predictions = []

    for batch in tqdm(data_loader, desc="Evaluating", dynamic_ncols=True):
        text_input_ids = batch["text_input_ids"].to(config.device)
        text_attention_mask = batch["text_attention_mask"].to(config.device)
        text_token_type_ids = batch["text_token_type_ids"].to(config.device)
        label_input_ids = batch["label_input_ids"].to(config.device)
        label_attention_mask = batch["label_attention_mask"].to(config.device)
        label_token_type_ids = batch["label_token_type_ids"].to(config.device)
        labels = batch["labels"].to(config.device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                text_token_type_ids=text_token_type_ids,
                label_input_ids=label_input_ids,
                label_attention_mask=label_attention_mask,
                label_token_type_ids=label_token_type_ids,
            )

        outputs = outputs.to(torch.float32)
        probs = torch.sigmoid(outputs).cpu().numpy()
        predictions = probs >= threshold

        all_probs.extend(probs)
        all_predictions.extend(predictions)
        all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    if not os.path.exists(f"./log/rcv1_macro_all/"):
        os.makedirs(f"./log/rcv1_macro_all/")

    with open(f"./log/rcv1_macro_all/all_predictions.jsonl", "w") as f:
        for pred_vec, label_vec, prob_vec in zip(
            all_predictions, all_labels, all_probs
        ):
            predicted_labels = [id2label[i] for i, p in enumerate(pred_vec) if p]
            real_labels = [id2label[i] for i, l in enumerate(label_vec) if l == 1.0]
            rounded_probs = [round(float(p), 3) for p in prob_vec]
            entry = {
                "real": real_labels,
                "predict": predicted_labels,
                "prob": rounded_probs,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_parent_levels(data):
    """获取层级结构，避免循环引用的问题"""
    levels = []
    current_level = ["root"]
    processed_nodes = set()  # 用于追踪已处理的节点

    while current_level:
        levels.append(current_level.copy())  # 保存当前层级
        next_level = []

        for node in current_level:
            if node in processed_nodes:  # 避免重复处理
                continue

            processed_nodes.add(node)

            if node in data:
                for child in data[node]:
                    if child in data and child not in processed_nodes:
                        next_level.append(child)

        current_level = next_level

    return levels


def main():
    parser = argparse.ArgumentParser(
        description="Test classifiers with specified dataset, augmentation flag and checkpoint name"
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

    start = time.time()

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
        torch.set_float32_matmul_precision("high")

    config = Config()
    if args.dataset:
        config.dataset_name = args.dataset
    if args.ckpt:
        config.checkpoint_subdir = args.ckpt
    config.enable_augmentation = args.aug
    base_dir = f"./checkpoints/{config.dataset_name}/{config.checkpoint_subdir}"

    with open(
        f"{config.data_path}/{config.dataset_name}/labels.json", "r", encoding="utf8"
    ) as f:
        all_labels = json.load(f)

    if config.enable_augmentation:
        with open(
            f"{config.data_path}/{config.dataset_name}/da_labels.json", "r", encoding="utf8"
        ) as f:
            da_labels_json = json.load(f)

        da_labels = []
        for key in da_labels_json.keys(): 
            da_labels.append(key)
    else:
        da_labels = []

    parent_levels = get_parent_levels(all_labels)

    for labels in parent_levels:
        for parent in labels:
            labels = all_labels[parent]
            print(f"Classifier: {parent}")

            if config.enable_augmentation:
                if parent not in da_labels:
                    continue

            _, _, test_loader, id2label = create_data_loaders(
                config,
                parent,
                labels,
            )
            print(f"Test iterations: {len(test_loader)}")

            model = BERTClassifier(config).to(config.device)
            model = torch.compile(model)

            model_path = os.path.join(
                base_dir, f"{parent.replace('/', '_')}/best_f1_micro_model.pt"
            )
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, weights_only=True))
                test(model, parent.replace("/", "_"), id2label, test_loader)

            print(f"Total time: {time.time() - start:.2f}s")

    print(f"Total time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
