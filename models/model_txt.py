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

import models
import utils
from .model_prompt import *

tokenizer = BartTokenizer.from_pretrained('bart-base')

class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False
class View(nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((self.shape))


def invert_mask(attention_mask):
    return attention_mask.eq(0)

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a input_ids with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)

def shift_tokens_right(input_ids, token_id):
    """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
        This is taken directly from modeling_bart.py
    """
    prev_output_tokens = input_ids.clone()
    #   index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = torch.ones_like(prev_output_tokens[:, 0], device=prev_output_tokens.device) * token_id
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens



class PromptGeneratorv20(torch.nn.Module):
    """
    use hiddensize*2
    """
    def __init__(self, config) -> None:
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained('bart-base')
        self.bart_config = self.bart.config
        img_processer = timm.create_model("vit_base_patch32_224", pretrained=True,)
        self.img_emb_layer = img_processer.patch_embed

        config.n_patchs = 49
        self.config = config
        self.prompt_learner = PromptLearnerv2(self.config, self.bart_config.d_model)

        hidden_size = 768
        self.projhead1 = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(hidden_size, hidden_size*2)),
                ('bn1', nn.BatchNorm1d(hidden_size*2)),
                ('relu1', nn.ReLU()),
                ('fc2', nn.Linear(hidden_size*2, hidden_size)),
                ('bn2', BatchNorm1dNoBias(hidden_size)),
            ]))
        

        self.extra_layers = models.GateExtractationLayers(self.bart_config)
        self.gate_W = nn.Linear(self.bart_config.d_model*2, self.bart_config.d_model)
        self.pjoj_gate = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(hidden_size, hidden_size)),
                ('bn1', nn.LayerNorm(hidden_size)),
                ('relu1', nn.ReLU()),
                ('fc2', nn.Linear(hidden_size, 1)),
            ]))
    
    def get_img_emb(self, img):
        img_feature = self.img_emb_layer(img)
        self.prompt_learner.init_token(img_feature, self.bart.model.encoder.embed_tokens)
        prompts = self.prompt_learner()
        embed_pos = self.bart.model.encoder.embed_positions(prompts.size())
        hidden_states = prompts+embed_pos
        return hidden_states
    
    def get_txt_emb(self, input_ids):
        input_txt_emb = self.bart.model.encoder.embed_tokens(input_ids)*self.bart.model.encoder.embed_scale
        embed_pos = self.bart.model.encoder.embed_positions(input_txt_emb.size())
        hidden_states = input_txt_emb+embed_pos

        return hidden_states
    
    def get_extra_feature(self, input_ids, attention_mask, img, inter_ids, use_GT=False):
        input_img_emb = self.img_emb_layer(img)
        embed_pos = self.bart.model.encoder.embed_positions(input_img_emb.size())
        hidden_states_img = input_img_emb+embed_pos
        encoder_img = self.bart.model.encoder(
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=hidden_states_img,
            output_attentions=self.bart_config.output_attentions,
            output_hidden_states=(self.bart_config.output_hidden_states),
            return_dict=None,
        )[0]
        encoder_txt = self.bart.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=self.bart_config.output_attentions,
            output_hidden_states=(self.bart_config.output_hidden_states),
            return_dict=None,
        )[0]
        residual = encoder_txt
        hidden_states = self.extra_layers(residual, encoder_img)
        gate_mask = self.pjoj_gate((self.gate_W(torch.cat([residual, hidden_states], dim=-1)))).squeeze()
        hard_gate_mask = (F.sigmoid(gate_mask)>self.config.theta).float()
        if use_GT:
            hard_gate_mask = inter_ids
        extra_feature_ids = utils.del_tensor_0_nodim(hard_gate_mask*input_ids.clone(), max_len=50, pad_token_id=tokenizer.pad_token_id)
        extra_feature_att = torch.ones_like(extra_feature_ids)*(extra_feature_ids!=tokenizer.pad_token_id)
        extra_feature = self.bart.model.encoder(
            input_ids=extra_feature_ids,
            attention_mask=extra_feature_att,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=self.bart_config.output_attentions,
            output_hidden_states=(self.bart_config.output_hidden_states),
            return_dict=None,
        )[0]

        return extra_feature, gate_mask


    def forward(self, input_ids, attention_mask, decoder_ids, decoder_attention_mask, img, inter_ids):
        # print('decoder_ids: ', decoder_ids)
        decoder_input_ids = shift_tokens_right(decoder_ids, self.bart_config.bos_token_id)
        # print("decoder_input_ids: ", decoder_input_ids)

        hidden_states_txt = self.get_txt_emb(input_ids)
        hidden_states_img = self.get_img_emb(img)

        encoder_1 = self.bart.model.encoder(
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=hidden_states_img,
            output_attentions=self.bart_config.output_attentions,
            output_hidden_states=(self.bart_config.output_hidden_states),
            return_dict=None,
        )[0]
        encoder_2 = self.bart.model.encoder(
            input_ids=None,
            attention_mask=attention_mask,
            head_mask=None,
            inputs_embeds=hidden_states_txt,
            output_attentions=self.bart_config.output_attentions,
            output_hidden_states=(self.bart_config.output_hidden_states),
            return_dict=None,
        )[0]

        extra_feature, gate_mask = self.get_extra_feature(input_ids, attention_mask, img, inter_ids, use_GT=False)

        # print("extra_feature: ", extra_feature.size())

        last_hidden_state = torch.cat([encoder_2, extra_feature, encoder_1], dim=1)
        # last_hidden_state = torch.cat([encoder_1[0], encoder_2[0]], dim=1)
        encoder_outputs = ModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=None,
            attentions=None,
        )

        with torch.no_grad():
            encoder_tgt = self.bart.model.encoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
            )[0]
            target_feature = encoder_tgt[:,0,:].detach()
        
        
        outputs = self.bart(
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            output_hidden_states=False,
            return_dict=True,
        )
        return outputs.logits, self.projhead1(encoder_1[:,0,:]), self.projhead1(target_feature), gate_mask
            
    
    @torch.no_grad()
    def generate_text(self, input_ids, attention_mask, img, inter_ids, beam_size=10):
        hidden_states_txt = self.get_txt_emb(input_ids)
        hidden_states_img = self.get_img_emb(img)
        hidden_states_txt_copy = hidden_states_txt.clone()

        encoder_1 = self.bart.model.encoder(
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=hidden_states_img,
            output_attentions=self.bart_config.output_attentions,
            output_hidden_states=(self.bart_config.output_hidden_states),
            return_dict=None,
        )[0]
        encoder_2 = self.bart.model.encoder(
            input_ids=None,
            attention_mask=attention_mask,
            head_mask=None,
            inputs_embeds=hidden_states_txt,
            output_attentions=self.bart_config.output_attentions,
            output_hidden_states=(self.bart_config.output_hidden_states),
            return_dict=None,
        )[0]


        extra_feature, gate_mask = self.get_extra_feature(input_ids, attention_mask, img, inter_ids, use_GT=False)

        # print("extra_feature: ", extra_feature.size())

        last_hidden_state = torch.cat([encoder_2, extra_feature, encoder_1], dim=1)
        # last_hidden_state = torch.cat([encoder_1[0], encoder_2[0]], dim=1)
        # print("last_hidden_state: ", last_hidden_state.size())
        encoder_outputs = ModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=None,
            attentions=None,
        )

        generate_ids = self.bart.generate(
            input_ids=None,
            attention_mask=None,
            use_cache=True,
            decoder_start_token_id=self.bart_config.bos_token_id, num_beams=beam_size, max_length=25,
            early_stopping=True,
            encoder_outputs=encoder_outputs,
        )

        ans = [tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in generate_ids]
        for i, s_a in enumerate(ans):
            if len(s_a)<=0:
                ans[i]="<empty>"

        # print(ans)
        # exit(0)
        return ans, gate_mask
