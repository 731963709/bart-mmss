import os
from collections import OrderedDict
from copy import deepcopy

import clip
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
from transformers import (BartConfig, BartForConditionalGeneration,
                          BartTokenizer)
from transformers.modeling_outputs import ModelOutput

tokenizer = BartTokenizer.from_pretrained('bart-base')

class PromptLearnerv2(nn.Module):
    """
        add proj head function
    """
    def __init__(self, cfg, emb_dim):
        super().__init__()
        n_cls = cfg.n_patchs
        n_ctx = cfg.n_prompt
        ctx_dim = emb_dim

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim
        self.batch_size = cfg.batch_size

        ctx_vectors = torch.empty(self.n_cls, self.n_ctx, self.ctx_dim)

        print(f"Number of context words (tokens): {self.n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.projhead = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.ctx_dim, self.ctx_dim)),
            ('bn1', nn.LayerNorm(self.ctx_dim)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(self.ctx_dim, self.ctx_dim)),
            ('bn2', nn.LayerNorm(self.ctx_dim)),
        ]))

        for p in self.parameters():
            p.data.normal_(-0.04, 0.04)

    def init_token(self, features, bart_emb):
        # random initialization
        with torch.no_grad():
            cls_token_tensor = torch.LongTensor([tokenizer.cls_token_id for i in range(features.size(0))])
            cls_token_tensor = cls_token_tensor.to(features.device)
            # print(cls_token_tensor.size())
            cls_embedding = bart_emb(cls_token_tensor)
            cls_embedding = cls_embedding.unsqueeze(1)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", cls_embedding)
        self.register_buffer("token_suffix", features)
    

    def forward(self):
        ctx = self.projhead(self.ctx)

        suffix = self.token_suffix
        prefix = self.token_prefix
        ctx = ctx.unsqueeze(0).expand(suffix.size(0), -1, -1, -1)
        # print(prefix.size())
        prompts = [prefix,]
        for i in range(self.n_cls):
            suffix_i = suffix[:, i : i + 1, :]
            ctx_i = ctx[:, i,:,:]
            # print("suffix_i: ", suffix_i.size())
            # print("ctx_i: ", ctx_i.size())
            prompt = torch.cat(
                [
                    ctx_i,
                    suffix_i,
                ],
                dim=1,
            )
            # print(prompt.size())
            prompts.append(prompt)
        prompts = torch.cat(prompts, dim=1)

        return prompts

