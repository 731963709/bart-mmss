import json
import linecache
import os

import nltk
import numpy as np
import torch
import torch.utils.data
import torchvision
from nltk.corpus import stopwords
from PIL import Image

# def get_intersection(words_1, words_2):
#     words1_list = words_1.split()
#     words2_list = words_2.split()
#     output_list = []
#     for word in words1_list:
#         if word in words2_list:
#             output_list.append(word)
#             output_list.append("</s>")
#     return " ".join(output_list)

def get_intersection(words_1, words_2):
    words1_list = words_1.split()
    words2_list = words_2.split()
    # output_list = ["</s>",]
    output_list = [" "]
    for word in words1_list:
        # if word in set(stopwords.words('english')):continue
        # if "#" in word:continue
        # if "'s" in word:continue
        # if "<unk>" in word:continue
        if word in words2_list:
            output_list.append(word)
            # output_list.append("</s>")
    return " ".join(output_list)

def get_intersection_idsmask(words_1, words_2, pad_token_id=1):
    output_ids_mask = torch.zeros_like(words_1, dtype=torch.float)
    for i, word1 in enumerate(words_1):
        for ids, w in enumerate(word1):
            if w==pad_token_id:continue
            if w in words_2[i]:
                output_ids_mask[i,ids] = 1

    return output_ids_mask

def get_intersection_ids(words_1, words_2, max_len=30, pad_token_id=1):
    ans = torch.ones((words_1.shape[0], max_len), device=words_1.device, dtype=words_1.dtype)*pad_token_id
    for i, word1 in enumerate(words_1):
        now_len=0
        for ids, w in enumerate(word1):
            if w==pad_token_id:continue
            if w in words_2[i]:
                ans[i,now_len] = w
                now_len +=1
    return ans


class BiDataset(torch.utils.data.Dataset):
    def __init__(self, infos, img_path, train=True):
        self.original_srcF = infos+"_sent.txt"
        self.original_tgtF = infos+"_title.txt"
        self.img_path = img_path
        self.train = train
        self.transform = torchvision.transforms.Compose([
                                torchvision.transforms.Resize(256),
                                torchvision.transforms.CenterCrop(224),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])
                            ])
        self.original_src = []
        self.original_tgt = []
        self.augment_src = []
        with open(self.original_srcF, "r") as f:
            for i in f:
                self.original_src.append(i.strip())
        with open(self.original_tgtF, "r") as f:
            for i in f:
                self.original_tgt.append(i.strip())
        

    def __getitem__(self, index):
        img = Image.open(
            os.path.join(self.img_path, f"{index+1}.jpg")
        ).convert("RGB")
        simg = self.transform(img)
        return self.original_src[index], self.original_tgt[index], simg

    def __len__(self):
        return len(self.original_src)

class BiDatasetv2(torch.utils.data.Dataset):
    def __init__(self, infos, img_path, train=True):
        self.original_srcF = infos+"_sent.txt"
        self.original_tgtF = infos+"_title.txt"
        self.img_path = img_path
        self.train = train
        self.transform = torchvision.transforms.Compose([
                                torchvision.transforms.Resize(256),
                                torchvision.transforms.CenterCrop(224),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])
                            ])
        self.original_src = []
        self.original_tgt = []
        self.intersection = []
        with open(self.original_srcF, "r") as f:
            for i in f:
                self.original_src.append(i.strip())
        with open(self.original_tgtF, "r") as f:
            for i in f:
                self.original_tgt.append(i.strip())


    def __getitem__(self, index):
        img = Image.open(
            os.path.join(self.img_path, f"{index+1}.jpg")
        ).convert("RGB")
        simg = self.transform(img)
        return self.original_src[index], self.original_tgt[index], simg, get_intersection(self.original_src[index], self.original_tgt[index])

    def __len__(self):
        return len(self.original_src)

class BiDataset_t(torch.utils.data.Dataset):
    def __init__(self, infos,):
        self.original_srcF = infos+"_sent.txt"
        self.original_tgtF = infos+"_title.txt"
        self.original_src = []
        self.original_tgt = []
        self.augment_src = []
        with open(self.original_srcF, "r") as f:
            for i in f:
                self.original_src.append(i.strip())
        with open(self.original_tgtF, "r") as f:
            for i in f:
                self.original_tgt.append(i.strip())
        

    def __getitem__(self, index):
        return self.original_src[index], self.original_tgt[index]

    def __len__(self):
        return len(self.original_src)

class BiDataset_tv2(torch.utils.data.Dataset):
    def __init__(self, infos,):
        self.original_srcF = infos+"_sent.txt"
        self.original_tgtF = infos+"_title.txt"
        self.original_src = []
        self.original_tgt = []
        self.augment_src = []
        with open(self.original_srcF, "r") as f:
            for i in f:
                self.original_src.append(i.strip())
        with open(self.original_tgtF, "r") as f:
            for i in f:
                self.original_tgt.append(i.strip())
        
    def __getitem__(self, index):
        src = self.original_src[index]
        tgt = self.original_tgt[index]
        return src+" "+get_intersection(src, tgt), tgt

    def __len__(self):
        return len(self.original_src)

def collate_function(data):
    src, tgts, simg = zip(*data)
    return (src, tgts, simg)

def collate_functionv2(data):
    src, tgts, simg, inter = zip(*data)
    return (src, tgts, simg, inter)

def collate_function_t(data):
    src, tgts = zip(*data)
    return (src, tgts)

def token_img(data_img):
    img_tensor = torch.zeros((len(data_img),)+data_img[0].shape)
    for i,s in enumerate(data_img):
        img_tensor[i]=data_img[i]
    return img_tensor


def token_seq(data_src, data_tgt, tokenizer, max_src_len=60, max_tgt_len=25, eval=True):
    token_sent = tokenizer.batch_encode_plus(data_src, max_length=max_src_len, padding=True, truncation=True, return_tensors='pt', add_special_tokens=False)
    if eval:
        return token_sent['input_ids'], token_sent['attention_mask'], \
            None, None
    # token_title = tokenizer.batch_encode_plus(data_tgt, max_length=25, pad_to_max_length=True, padding=True, truncation=True, return_tensors='pt', add_special_tokens=False)
    all_tokens = []
    all_mask = []
    for text in data_tgt:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) >= max_tgt_len:
            tokens = tokens[:max_tgt_len-1]
        tokens = tokens + [tokenizer.eos_token_id]
        mask = [1] * len(tokens) + [0] * (max_tgt_len - len(tokens))
        tokens = tokens + [tokenizer.pad_token_id] * (max_tgt_len - len(tokens))
        all_tokens.append(tokens)
        all_mask.append(mask)

    return token_sent['input_ids'], token_sent['attention_mask'], \
            torch.tensor(all_tokens), torch.tensor(all_mask)


def token_seq_t5(data_src, data_tgt, tokenizer, eval=True):
    if eval:
        add_data_src = []
        for src in data_src:
            add_data_src.append("summarize: " + src)
        token_sent = tokenizer.batch_encode_plus(add_data_src, max_length=70, pad_to_max_length=True, padding='max_length', truncation=True, return_tensors='pt', add_special_tokens=False)
        return token_sent['input_ids'], token_sent['attention_mask'], \
            None, None
    token_sent = tokenizer.batch_encode_plus(data_src, max_length=70, pad_to_max_length=True, padding='max_length', truncation=True, return_tensors='pt', add_special_tokens=False)
    # token_title = tokenizer.batch_encode_plus(data_tgt, max_length=25, pad_to_max_length=True, padding='max_length', truncation=True, return_tensors='pt', add_special_tokens=False)
    all_tokens = []
    all_mask = []
    tgt_max_len = 25
    for text in data_tgt:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) >= tgt_max_len:
            tokens = tokens[:tgt_max_len-1]
        tokens = tokens + [tokenizer.eos_token_id]
        mask = [1] * len(tokens) + [0] * (tgt_max_len - len(tokens))
        tokens = tokens + [tokenizer.pad_token_id] * (tgt_max_len - len(tokens))
        all_tokens.append(tokens)
        all_mask.append(mask)

    return token_sent['input_ids'], token_sent['attention_mask'], \
            torch.tensor(all_tokens), torch.tensor(all_mask)



def del_tensor_0_cloumn(Cs):
    idx = torch.where(torch.all(Cs[..., :] == 0, axis=0))[0]
    all = torch.arange(Cs.shape[1]).to(Cs.device)
    for i in range(len(idx)):
        all = all[torch.arange(all.size(0), device=Cs.device)!=idx[i]-i] 
    Cs = torch.index_select(Cs, 1, all)
    return Cs

def del_tensor_0(Cs, max_len=30):
    bacth_size, Cs_len, dim = Cs.shape
    ans = torch.zeros((bacth_size, max_len, dim), device=Cs.device)
    for b, CC in enumerate(Cs):
        now_len=0
        for cc in CC:
            if cc[0]>0.5:
                ans[b,now_len,:]=cc
                now_len+=1
    return ans

def del_tensor_0_nodim(Cs, max_len=30, pad_token_id=1):
    bacth_size, Cs_len = Cs.shape
    ans = torch.ones((bacth_size, max_len), device=Cs.device, dtype=torch.long)*pad_token_id
    for b, CC in enumerate(Cs):
        now_len=0
        # print(CC)
        for cc in CC:
            if cc>0.5:
                ans[b,now_len]=cc
                now_len+=1
                if now_len >= max_len:break
    return ans