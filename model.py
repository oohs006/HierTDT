import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from cleam_bert import EnhancedBertModel


class MHXAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config.num_labels
        self.words_per_label = config.words_per_label
        self.num_heads = config.num_heads
        self.head_size = config.hidden_size // config.num_heads
        self.all_head_size = self.num_heads * self.head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, source, target, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(source))
        key_layer = self.transpose_for_scores(self.key(target))
        value_layer = self.transpose_for_scores(self.value(target))
        value_layer = value_layer.view(
            value_layer.size()[:-2]
            + (self.num_classes, self.words_per_label, self.head_size)
        )

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-2, -1))

        if attention_mask is not None:
            attention_mask = torch.where(
                attention_mask == 0,
                torch.tensor(
                    -10000, dtype=attention_scores.dtype, device=attention_scores.device
                ),
                torch.zeros(
                    1, dtype=attention_scores.dtype, device=attention_scores.device
                ),
            )
            attention_scores = attention_scores + attention_mask

        attention_scores = attention_scores.view(
            attention_scores.size()[:-2] + (self.num_classes, 1, self.words_per_label)
        )
        attention_probs = F.softmax(attention_scores, dim=-1)

        context_layer = torch.matmul(attention_probs, value_layer).squeeze(-2)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(
            context_layer.size()[:-2] + (self.all_head_size,)
        )

        return context_layer


class BERTClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = EnhancedBertModel.from_pretrained("bert-base-uncased")
        self.bert.embeddings_ = copy.deepcopy(self.bert.embeddings)
        self.bert.encoder.layer_ = copy.deepcopy(self.bert.encoder.layer)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.mhxatt = MHXAttention(config)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        text_input_ids,
        text_attention_mask,
        text_token_type_ids,
        label_input_ids,
        label_attention_mask,
        label_token_type_ids,
    ):
        text_outputs, label_outputs = self.bert(
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            text_token_type_ids=text_token_type_ids,
            label_input_ids=label_input_ids,
            label_attention_mask=label_attention_mask,
            label_token_type_ids=label_token_type_ids,
        )

        text_cls = text_outputs[:, 0].unsqueeze(1)
        label_words = label_outputs[:, 1:-1]
        label_attention = label_attention_mask[:, 1:-1].view(
            label_attention_mask.size(0), 1, 1, -1
        )

        label_features = self.mhxatt(text_cls, label_words, label_attention)
        label_features = self.LayerNorm(label_features)
        logits = self.classifier(label_features).squeeze(-1)

        return logits
