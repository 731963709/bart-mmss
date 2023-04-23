import math
from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bart.modeling_bart import *


class Multihead_Attention(nn.Module):
    """
    Multi-head Attention
    """

    def __init__(self,
        model_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,):
        """
        initialization for variables and functions
        :param model_dim: hidden size
        :param num_heads: head number, default 8
        :param dropout: dropout probability
        """
        super(Multihead_Attention, self).__init__()

        self.head_dim = model_dim // num_heads
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.linear_keys = nn.Linear(model_dim, num_heads * self.head_dim, bias=bias)
        self.linear_values = nn.Linear(model_dim, num_heads * self.head_dim, bias=bias)
        self.linear_query = nn.Linear(model_dim, num_heads * self.head_dim, bias=bias)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)
        self.sigmoid = nn.Hardtanh(min_val=0)

    def forward(self, hidden_states, key_value_states, return_top_attn=False):
        """
        run multi-head attention
        :param key: key, [batch, len, size]
        :param value: value, [batch, len, size]
        :param query: query, [batch, len, size]
        :param mask: mask
        :param layer_cache: layer cache for transformer decoder
        :param type: "self" or "context"
        :param tau: temperature, will be deprecated
        :param Bernoulli: use Bernoulli selection or not
        :return: attention output and attention weights
        """
        query = hidden_states
        key = value = key_value_states
        
        batch_size = key.size(0)
        head_dim = self.head_dim
        head_count = self.num_heads
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, head_dim) \
                .transpose(1, 2)    # [batch, head, len, head_dim]

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * head_dim)    # [batch, len, size]

        # For transformer decoder.
        # denote the device for multi-gpus
        query, key, value = self.linear_query(query),\
            self.linear_keys(key),\
            self.linear_values(value)   # [batch, len, size]
        key = shape(key)    # [batch, head, k_len, head_dim]
        value = shape(value)    # [batch, head, v_len, head_dim]

        query = shape(query)    # [batch, head, q_len, head_dim]

        key_len = key.size(2)
        query_len = query.size(2)

        query = query / math.sqrt(head_dim)

        scores = torch.matmul(query, key.transpose(2, 3))   # [batch, head, q_len, k_len]


        # use Bernoulli selection or not
        attn = self.softmax(scores)  # [batch, head, q_len, k_len]

        drop_attn = self.dropout(attn)  # [batch, head, q_len, k_len]
        context = unshape(torch.matmul(drop_attn, value))   # [batch, q_len, size]

        output = self.final_linear(context)  # [batch, q_len, size]

        top_attn = attn \
            .view(batch_size, head_count,
                  query_len, key_len)[:, 0, :, :] \
            .contiguous()   # [batch, q_len, k_len]
        if return_top_attn:
            return output, top_attn
        return output

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model
        self.dropout = 0.1
        
        self.self_attn = Multihead_Attention(config.d_model, config.decoder_attention_heads, 0.1)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.cross_attn = Multihead_Attention(config.d_model, config.decoder_attention_heads, 0.1)
        self.cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, 3072)
        self.fc2 = nn.Linear(3072, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = F.gelu
        self.activation_dropout = 0.1
    
    def forward(self, hidden_states, key_value_states):
        # self-attn
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states, hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # cross-attn
        residual = hidden_states
        hidden_states = self.cross_attn(hidden_states, key_value_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.cross_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class GateExtraLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model
        self.dropout = 0.1
        self.self_attn = Multihead_Attention(config.d_model, config.decoder_attention_heads, 0.1)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.cross_attn = Multihead_Attention(config.d_model, config.decoder_attention_heads, 0.1)
        self.cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.gate_W = nn.Linear(config.d_model*2, config.d_model)

        self.fc1 = nn.Linear(self.embed_dim, 3072)
        self.fc2 = nn.Linear(3072, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = F.gelu
        self.activation_dropout = 0.1
    
    def forward(self, hidden_states, key_value_states, return_top_attn=False):
        # self-attn
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states, hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # cross-attn
        residual = hidden_states
        if return_top_attn:
            hidden_states, top_attn = self.cross_attn(hidden_states, key_value_states, return_top_attn=True)
        else:
            hidden_states = self.cross_attn(hidden_states, key_value_states, return_top_attn=False)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        # gate
        gate_mask = F.sigmoid(self.gate_W(torch.cat([residual, hidden_states], dim=-1)))
        hidden_states = gate_mask * hidden_states
        hidden_states = residual + hidden_states
        hidden_states = self.cross_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if return_top_attn:
            return hidden_states, top_attn
        return hidden_states


class VG_Multihead_Attention(nn.Module):
    def __init__(self,
        model_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,):
        """
        initialization for variables and functions
        :param model_dim: hidden size
        :param num_heads: head number, default 8
        :param dropout: dropout probability
        """
        super().__init__()

        self.head_dim = model_dim // num_heads
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.linear_keys = nn.Linear(model_dim, num_heads * self.head_dim, bias=bias)
        self.linear_values = nn.Linear(model_dim, num_heads * self.head_dim, bias=bias)
        self.linear_query = nn.Linear(model_dim, num_heads * self.head_dim, bias=bias)
        self.sigmoid = nn.Hardtanh(min_val=0)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)
        for p in self.parameters():
                p.data.uniform_(-0.5, 0.5)

    def forward(self, hidden_states, key_value_states, return_top_attn=False):
        """
        run multi-head attention
        :param key: key, [batch, len, size]
        :param value: value, [batch, len, size]
        :param query: query, [batch, len, size]
        :param mask: mask
        :param layer_cache: layer cache for transformer decoder
        :param type: "self" or "context"
        :param tau: temperature, will be deprecated
        :param Bernoulli: use Bernoulli selection or not
        :return: attention output and attention weights
        """
        query = hidden_states
        key = value = key_value_states
        
        batch_size = key.size(0)
        head_dim = self.head_dim
        head_count = self.num_heads
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, head_dim) \
                .transpose(1, 2)    # [batch, head, len, head_dim]

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * head_dim)    # [batch, len, size]

        # For transformer decoder.
        # denote the device for multi-gpus
        query, key, value = self.linear_query(query),\
            self.linear_keys(key),\
            self.linear_values(value)   # [batch, len, size]
        key = shape(key)    # [batch, head, k_len, head_dim]
        value = shape(value)    # [batch, head, v_len, head_dim]

        query = shape(query)    # [batch, head, 1, head_dim]

        key_len = key.size(2)
        query_len = query.size(2)

        query = query / math.sqrt(head_dim)

        scores = torch.matmul(query, key.transpose(2, 3)).squeeze()    # [batch, head, k_len]


        # use Bernoulli selection or not
        attn = self.sigmoid(scores) # [batch, head, k_len]

        drop_attn = self.dropout(attn)  # [batch, head, k_len]
        # print("drop_attn: ", drop_attn.size())
        # print("value: ", value.size())
        context = unshape(drop_attn.unsqueeze(-1).repeat(1,1,1,head_dim)*value)   # [batch, k_len, size]

        output = self.final_linear(context)  # [batch, k_len, size]

        top_attn = scores \
            .view(batch_size, head_count,
                  key_len)[:, 0, :] \
            .contiguous()   # [batch, k_len]
        mean_attn = torch.mean(scores.view(batch_size, head_count, key_len).contiguous(), dim=1) # [batch, k_len]
        if return_top_attn:
            return output, mean_attn
        return output

# class ExtractationLayers(nn.Module):
#     def __init__(self, bart_config):
#         super().__init__()
#         self.ori_bart = BartModel.from_pretrained('bart-base')
#         num_of_layers = 2
#         print("num_of_layers: ", num_of_layers)
#         self.layers = self.ori_bart.decoder.layers[0:num_of_layers]
    
#     def forward(self, hidden_states, key_value_states):
#         # attention_mask = torch.ones()
#         for idx, decoder_layer in enumerate(self.layers):
#             layer_outputs = decoder_layer(
#                     hidden_states,
#                     attention_mask=None,
#                     encoder_hidden_states=key_value_states,
#                     encoder_attention_mask=None,
#                 )
#             hidden_states = layer_outputs[0]
#         return hidden_states

class ExtractationLayers(nn.Module):
    def __init__(self, bart_config):
        super().__init__()
        new_config = deepcopy(bart_config)
        new_config.decoder_layers = 3
        print("decoder_layers: ", new_config.decoder_layers)
        new_config.decoder_attention_heads = 8
        print("decoder_attention_heads: ", new_config.decoder_attention_heads)
        self.config = new_config
        self.layers = nn.ModuleList([TransformerDecoderLayer(self.config) for _ in range(self.config.decoder_layers)])

        for p in self.parameters():
            p.data.uniform_(-0.2, 0.2)
    
    def forward(self, hidden_states, key_value_states):
        for idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(hidden_states, key_value_states)
        return hidden_states


class GateExtractationLayers(nn.Module):
    def __init__(self, bart_config):
        super().__init__()
        new_config = deepcopy(bart_config)
        new_config.decoder_layers = 3
        print("decoder_layers: ", new_config.decoder_layers)
        new_config.decoder_attention_heads = 8
        print("decoder_attention_heads: ", new_config.decoder_attention_heads)
        self.config = new_config
        self.layers = nn.ModuleList([GateExtraLayer(self.config) for _ in range(self.config.decoder_layers)])

        for p in self.parameters():
            p.data.uniform_(-0.2, 0.2)
    
    def forward(self, hidden_states, key_value_states, return_top_attn=False):
        if return_top_attn:
            for idx, decoder_layer in enumerate(self.layers):
                hidden_states, top_attn = decoder_layer(hidden_states, key_value_states, return_top_attn=True)
            return hidden_states, top_attn
        for idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(hidden_states, key_value_states, return_top_attn=False)
        return hidden_states


