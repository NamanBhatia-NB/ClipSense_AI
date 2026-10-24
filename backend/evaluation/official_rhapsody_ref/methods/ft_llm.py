import torch
import torch.nn as nn
import numpy as np
import yaml
import os
from methods.base import BaseMethod
from utils import graph_to_indices


def build_input(segment_summaries, seg_end_token):
    """Concatenate segment summaries with end-of-segment token after each."""
    parts = []
    for summary in segment_summaries:
        parts.append(str(summary) + seg_end_token)
    return ''.join(parts)


def build_binary_labels(gt, max_n_gt, gt_threshold):
    """Convert replay scores to binary labels using graph_to_indices."""
    indices = graph_to_indices(gt, max_n_gt, gt_threshold)
    labels = torch.zeros(len(gt), dtype=torch.float32)
    for idx in indices:
        labels[idx] = 1.0
    return labels


class SegmentClassifier(nn.Module):
    """Linear classifier for per-segment binary classification."""
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


class AudioProjection(nn.Module):
    """Dropout + Linear projection + ReLU for audio features."""
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(self.dropout(x)))


def predict_single(model, tokenizer, cls_head, audio_proj, row, config, seg_end_token_id, device):
    """Run inference on a single row and return predicted highlight indices."""
    text = build_input(row['segment_summaries'], config['seg_end_token'])
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=config['max_seq_length'],
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    last_hidden = outputs.hidden_states[-1]
    seg_positions = (inputs['input_ids'][0] == seg_end_token_id).nonzero(as_tuple=False)
    segment_reprs = last_hidden[0, seg_positions[:, 0], :]

    # audio fusion
    if config.get('use_dva', False) and audio_proj is not None:
        audio_feat = torch.tensor(np.array(row['dva']), dtype=torch.float32).to(device)
        audio_feat = audio_proj(audio_feat)
        segment_reprs = torch.cat([segment_reprs, audio_feat], dim=-1)
    elif config.get('use_hubert', False) and audio_proj is not None:
        hubert_feat = torch.tensor(
            np.array(row['hubert'][:, config.get('hubert_layer', 9), :]),
            dtype=torch.float32,
        ).to(device)
        hubert_feat = audio_proj(hubert_feat)
        segment_reprs = torch.cat([segment_reprs, hubert_feat], dim=-1)

    probs = torch.sigmoid(cls_head(segment_reprs.float())).squeeze(-1)

    mask = probs > config['threshold']
    if mask.sum() == 0:
        return []

    indices = mask.nonzero(as_tuple=False).squeeze(-1)
    sorted_order = probs[indices].argsort(descending=True)
    result = indices[sorted_order].cpu().tolist()
    return result[:config.get('max_n_gt', 20)]


def print_trainable_params(model, cls_head, audio_proj=None):
    """Count and print trainable parameters."""
    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cls_params = sum(p.numel() for p in cls_head.parameters())
    audio_params = sum(p.numel() for p in audio_proj.parameters()) if audio_proj is not None else 0
    total = lora_params + cls_params + audio_params
    print(f"LoRA parameters: {lora_params:,}")
    print(f"Classifier parameters: {cls_params:,}")
    if audio_proj is not None:
        print(f"Audio projection parameters: {audio_params:,}")
    print(f"Total trainable parameters: {total:,}")


class FtLlmMethod(BaseMethod):
    """Fine-tuned LLM with QLoRA and segment-level classification."""

    @classmethod
    def add_cli_args(cls, parser):
        parser.add_argument(
            "--checkpoint_path",
            type=str,
            required=True,
            help="path to saved checkpoint directory",
        )
        parser.add_argument(
            "--config",
            type=str,
            default=None,
            help="optional YAML config override",
        )
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            "--use_dva",
            action="store_true",
            help="use DVA audio features",
        )
        group.add_argument(
            "--use_hubert",
            action="store_true",
            help="use HuBERT audio features",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=None,
            help="prediction threshold override",
        )

    def __init__(self, args, train_dataset):
        from unsloth import FastLanguageModel

        # load config from checkpoint dir
        config_path = os.path.join(args.checkpoint_path, 'ft_llm_config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # override with CLI args
        if args.use_dva:
            self.config['use_dva'] = True
            self.config['use_hubert'] = False
        if args.use_hubert:
            self.config['use_hubert'] = True
            self.config['use_dva'] = False
        if args.threshold is not None:
            self.config['threshold'] = args.threshold
        if args.config is not None:
            with open(args.config, 'r') as f:
                override = yaml.safe_load(f)
            self.config.update(override)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = self.config['threshold']
        self.use_dva = self.config.get('use_dva', False)
        self.use_hubert = self.config.get('use_hubert', False)
        self.hubert_layer = self.config.get('hubert_layer', 9)
        self.max_n_gt = self.config.get('max_n_gt', 20)
        self.seg_end_token = self.config['seg_end_token']

        # load model and tokenizer
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config['model_name'],
            max_seq_length=self.config['max_seq_length'],
            load_in_4bit=True,
        )

        # apply LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.config['lora_r'],
            lora_alpha=self.config['lora_alpha'],
            lora_dropout=self.config['lora_dropout'],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

        # handle end-of-segment token
        if self.seg_end_token not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({'additional_special_tokens': [self.seg_end_token]})
            model.resize_token_embeddings(len(tokenizer))
        self.seg_end_token_id = tokenizer.convert_tokens_to_ids(self.seg_end_token)

        # determine classifier input dim
        hidden_dim = 2048
        if self.use_dva:
            cls_input_dim = hidden_dim + 1024
        elif self.use_hubert:
            cls_input_dim = hidden_dim + 768
        else:
            cls_input_dim = hidden_dim

        # create classification head
        self.cls_head = SegmentClassifier(cls_input_dim).to(self.device)

        # create audio projection if needed
        self.audio_proj = None
        if self.use_dva:
            self.audio_proj = AudioProjection(1024).to(self.device)
        elif self.use_hubert:
            self.audio_proj = AudioProjection(768).to(self.device)

        # load checkpoint weights
        adapter_path = args.checkpoint_path
        model.load_adapter(adapter_path, adapter_name="default")

        cls_head_path = os.path.join(args.checkpoint_path, 'cls_head.pt')
        self.cls_head.load_state_dict(torch.load(cls_head_path, map_location=self.device))

        if self.audio_proj is not None:
            audio_proj_path = os.path.join(args.checkpoint_path, 'audio_proj.pt')
            self.audio_proj.load_state_dict(torch.load(audio_proj_path, map_location=self.device))

        # set eval mode
        FastLanguageModel.for_inference(model)
        self.cls_head.eval()
        if self.audio_proj is not None:
            self.audio_proj.eval()

        self.model = model
        self.tokenizer = tokenizer

        print_trainable_params(model, self.cls_head, self.audio_proj)

    def predict(self, row):
        return predict_single(
            self.model, self.tokenizer, self.cls_head, self.audio_proj,
            row, self.config, self.seg_end_token_id, self.device,
        )
