import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from transformers import get_linear_schedule_with_warmup
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning, module="torch.autograd.graph")


class Trainer:
    def __init__(
        self,
        model,
        config,
        log_file,
        classifier_name,
        train_loader,
        val_loader,
        test_loader=None,
    ):
        self.model = model
        self.config = config
        self.log_file = log_file
        self.classifier_name = classifier_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.best_metrics = {
            "loss": float("inf"),
            "f1_macro": float("-inf"),
            "f1_micro": float("-inf"),
        }

        self.best_epochs = {"loss": 0, "f1_macro": 0, "f1_micro": 0}

        self.optimizer, self.scheduler = self._configure_optimizer()
        self.criterion = nn.BCEWithLogitsLoss()

    def _configure_optimizer(self):
        param_dict = {
            pn: p for pn, p in self.model.named_parameters() if p.requires_grad
        }

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"Number of parameters with weight decay: {len(decay_params)}, Total parameters: {num_decay_params:,}"
        )
        print(
            f"Number of parameters without weight decay: {len(nodecay_params)}, Total parameters: {num_nodecay_params:,}"
        )

        try:
            test_param = torch.nn.Parameter(torch.randn(1, device=self.config.device))
            test_optimizer = torch.optim.AdamW([test_param], lr=1e-3, fused=True)
            use_fused = True
            del test_optimizer, test_param
        except RuntimeError:
            use_fused = False

        print(f"Using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.config.learning_rate, fused=use_fused
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=len(self.train_loader) * self.config.epochs * 0.1,
            num_training_steps=len(self.train_loader) * self.config.epochs,
        )

        return optimizer, scheduler

    def save_model(self, filename):
        os.makedirs(
            f"{self.config.checkpoint_dir}/{self.classifier_name}", exist_ok=True
        )

        path = os.path.join(
            f"{self.config.checkpoint_dir}/{self.classifier_name}", filename
        )
        torch.save(self.model.state_dict(), path)
        print(f"Saved model weights to {path}")

    def update_best_metrics(self, val_results, epoch):
        if val_results["loss"] < self.best_metrics["loss"]:
            self.best_metrics["loss"] = val_results["loss"]
            self.best_epochs["loss"] = epoch + 1
            self.save_model("best_loss_model.pt")
            print(f"New best loss: {val_results['loss']:.4f} at epoch {epoch + 1}")

        if val_results["f1_macro"] > self.best_metrics["f1_macro"]:
            self.best_metrics["f1_macro"] = val_results["f1_macro"]
            self.best_epochs["f1_macro"] = epoch + 1
            self.save_model("best_f1_macro_model.pt")
            print(
                f"New best F1 macro: {val_results['f1_macro']:.4f} at epoch {epoch + 1}"
            )

        if val_results["f1_micro"] > self.best_metrics["f1_micro"]:
            self.best_metrics["f1_micro"] = val_results["f1_micro"]
            self.best_epochs["f1_micro"] = epoch + 1
            self.save_model("best_f1_micro_model.pt")
            print(
                f"New best F1 micro: {val_results['f1_micro']:.4f} at epoch {epoch + 1}"
            )

    def train_epoch(self, epoch):
        self.model.train()

        max_steps = len(self.train_loader)

        for step, batch in enumerate(self.train_loader):
            t0 = time.time()
            last_step = step == max_steps - 1

            text_input_ids = batch["text_input_ids"].to(self.config.device)
            text_attention_mask = batch["text_attention_mask"].to(self.config.device)
            text_token_type_ids = batch["text_token_type_ids"].to(self.config.device)
            label_input_ids = batch["label_input_ids"].to(self.config.device)
            label_attention_mask = batch["label_attention_mask"].to(self.config.device)
            label_token_type_ids = batch["label_token_type_ids"].to(self.config.device)
            labels = batch["labels"].to(self.config.device)

            self.optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.model(
                    text_input_ids=text_input_ids,
                    text_attention_mask=text_attention_mask,
                    text_token_type_ids=text_token_type_ids,
                    label_input_ids=label_input_ids,
                    label_attention_mask=label_attention_mask,
                    label_token_type_ids=label_token_type_ids,
                )
                loss = self.criterion(outputs, labels)

            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            torch.cuda.synchronize()
            t1 = time.time()
            dt = t1 - t0
            batch_size = text_input_ids.size(0)
            seq_length = text_input_ids.size(1)
            tokens_processed = batch_size * seq_length
            tokens_per_sec = tokens_processed / dt
            dt_ms = dt * 1000

            if (step != 0 and step % self.config.train_interval == 0) or last_step:
                print(
                    f"step: {step:4d} | "
                    f"loss: {loss.item():6f} | "
                    f"lr: {current_lr:.2e} | "
                    f"norm: {norm:.4f} | "
                    f"time: {dt_ms:.2f}ms | "
                    f"tokens/sec: {tokens_per_sec:.2f}"
                )

                with open(self.log_file, "a") as f:
                    f.write(
                        f"step: {step:4d} | "
                        f"loss: {loss.item():6f} | "
                        f"lr: {current_lr:.2e} | "
                        f"norm: {norm:.4f} | "
                        f"time: {dt_ms:.2f}ms | "
                        f"tokens/sec: {tokens_per_sec:.2f}\n"
                    )

            if (step != 0 and step % self.config.eval_interval == 0) or last_step:
                val_results = self.evaluate(self.val_loader)
                print(
                    f"\nValidation results - "
                    f"loss: {val_results['loss']:.4f} | "
                    f"accuracy: {val_results['accuracy']:.4f} | "
                    f"f1_micro: {val_results['f1_micro']:.4f} | "
                    f"f1_macro: {val_results['f1_macro']:.4f}\n"
                )

                with open(self.log_file, "a") as f:
                    f.write(
                        f"Validation results - "
                        f"loss: {val_results['loss']:.4f} | "
                        f"accuracy: {val_results['accuracy']:.4f} | "
                        f"f1_micro: {val_results['f1_micro']:.4f} | "
                        f"f1_macro: {val_results['f1_macro']:.4f}\n"
                    )

                self.update_best_metrics(val_results, epoch)
                self.model.train()

    @torch.no_grad()
    def evaluate(self, data_loader):
        self.model.eval()
        all_labels = []
        all_predictions = []
        total_loss = 0

        for batch in tqdm(data_loader, desc="Evaluating", dynamic_ncols=True):
            text_input_ids = batch["text_input_ids"].to(self.config.device)
            text_attention_mask = batch["text_attention_mask"].to(self.config.device)
            text_token_type_ids = batch["text_token_type_ids"].to(self.config.device)
            label_input_ids = batch["label_input_ids"].to(self.config.device)
            label_attention_mask = batch["label_attention_mask"].to(self.config.device)
            label_token_type_ids = batch["label_token_type_ids"].to(self.config.device)
            labels = batch["labels"].to(self.config.device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.model(
                    text_input_ids=text_input_ids,
                    text_attention_mask=text_attention_mask,
                    text_token_type_ids=text_token_type_ids,
                    label_input_ids=label_input_ids,
                    label_attention_mask=label_attention_mask,
                    label_token_type_ids=label_token_type_ids,
                )

                loss = self.criterion(outputs, labels)

            total_loss += loss.item()

            outputs = outputs.to(torch.float32)
            predictions = torch.sigmoid(outputs).cpu().numpy() >= 0.5
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        results = {
            "loss": total_loss / len(data_loader),
            "labels": all_labels,
            "predictions": all_predictions,
            "accuracy": metrics.accuracy_score(all_labels, all_predictions),
            "f1_micro": metrics.f1_score(
                all_labels, all_predictions, average="micro", zero_division=0
            ),
            "f1_macro": metrics.f1_score(
                all_labels, all_predictions, average="macro", zero_division=0
            ),
        }

        return results

    def train(self):
        for epoch in range(self.config.epochs):
            print(f"\nEpoch: {epoch + 1}/{self.config.epochs}")

            with open(self.log_file, "a") as f:
                f.write(f"\nEpoch: {epoch + 1}/{self.config.epochs}\n")

            self.train_epoch(epoch)
            self.save_model(f"model_epoch_{epoch+1}.pt")

        print("\n=== Training Completed ===")
        print("\nBest Results Summary:")
        print(
            f"Best Loss: {self.best_metrics['loss']:.4f} (Epoch {self.best_epochs['loss']})"
        )
        print(
            f"Best F1 Macro: {self.best_metrics['f1_macro']:.4f} (Epoch {self.best_epochs['f1_macro']})"
        )
        print(
            f"Best F1 Micro: {self.best_metrics['f1_micro']:.4f} (Epoch {self.best_epochs['f1_micro']})"
        )

        with open(self.log_file, "a") as f:
            f.write("\n=== Training Completed ===\n")
            f.write("\nBest Results Summary:\n")
            f.write(
                f"Best Loss: {self.best_metrics['loss']:.4f} (Epoch {self.best_epochs['loss']})\n"
            )
            f.write(
                f"Best F1 Macro: {self.best_metrics['f1_macro']:.4f} (Epoch {self.best_epochs['f1_macro']})\n"
            )
            f.write(
                f"Best F1 Micro: {self.best_metrics['f1_micro']:.4f} (Epoch {self.best_epochs['f1_micro']})\n"
            )
