from dataclasses import dataclass
import torch

@dataclass
class Config:
    data_path = "data"
    dataset_name = "rcv1"
    max_length = 512
    train_batch_size = 24
    valid_batch_size = 24
    test_batch_size = 32
    epochs = 10
    eval_interval = 500
    train_interval = 50
    learning_rate = 3e-5
    weight_decay = 0.02
    model_name = "bert-base-uncased"
    num_labels = None
    checkpoint_subdir = "layering"
    dropout_rate = 0.3
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enable_augmentation = True

    num_hidden_layers = 12
    num_heads = 12
    hidden_size = 768
    words_per_label = 3

