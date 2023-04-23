import copy
import datetime
import os
import sys
import time
import warnings
from collections import Iterable, OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast
from tqdm import tqdm
from transformers import (BartConfig, BartForConditionalGeneration,
                          BartTokenizer)

import models
import utils
from dataset import load_datav2
from models.loss import NT_Xent

torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_num_threads(4)
warnings.filterwarnings("ignore")

LIMIT_NUM = 5

tokenizer = BartTokenizer.from_pretrained("bart-base")


def build_model(checkpoints, config):
    """
    build model, either Seq2Seq or Tensor2Tensor
    :param checkpoints: load checkpoint if there is pretrained model
    :return: model, optimizer and the print function
    """

    # model
    print(config)
    print("building model...\n")
    model = getattr(models, "PromptGeneratorv20")(config)
    model.to(config.device)
    if checkpoints is not None:
        model.load_state_dict(checkpoints["model"])

    # extra_layers_params = list(map(id, model.extra_layers.parameters()))
    # base_params = filter(lambda p: id(p) not in extra_layers_params, model.parameters())
    # params = [
    #     {"params": base_params , "lr": config.lr},
    #     {"params": model.extra_layers.parameters(), "lr": config.lr*100},
    # ]
    # optimizer = optim.Adam(params, lr=config.lr, betas=(config.beta1, config.beta2))
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, betas=(config.beta1, config.beta2))
    param_count = sum([param.view(-1).size()[0] for param in model.parameters()])
    # print(repr(model) + "\n\n")
    print("total number of model txt parameters: %d\n\n" % param_count)

    return model, optimizer,


def train_model(model_txt, data, optimz_txt, params, config, scaler):
    device = config.device
    model_txt.to(device)
    model_txt.train()
    train_loader = data["train_loader"]

    log_vars = defaultdict(float)
    return_dict = {}
    patience = num_trial = best_model_iter = 0
    num_correct = num_total = 0
    eval_updates = len(train_loader) //5
    # eval_updates = 100

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    # criterion = models.LabelSmoothingLoss(config.label_smoothing, tokenizer.vocab_size, ignore_index=tokenizer.pad_token_id)
    criterion.to(device)

    sim_loss = NT_Xent(config.sim_temp)
    sim_loss.to(device)

    bce_loss = nn.BCEWithLogitsLoss(reduction="none")
    bce_loss.to(device)

    mse_loss = nn.MSELoss(reduction="none")
    mse_loss.to(device)

    for i in range(1, config.epoch + 1):
        print("now epoch: ", i)
        for ori_src, ori_tgt, simg, ori_inter in train_loader:
            simg = utils.token_img(simg)
            src, src_att, tgt, tgt_att = utils.token_seq(ori_src, ori_tgt, tokenizer, eval=False)
            inter_ids = utils.get_intersection_idsmask(src, tgt)
            # print("tgt: ", tgt)

            src, tgt = src.to(device), tgt.to(device)
            src_att, tgt_att = src_att.to(device), tgt_att.to(device)
            simg = simg.to(device)
            inter_ids = inter_ids.to(device)

            optimz_txt.zero_grad()
            with autocast():
                # (batch_size, tgt_sent_len, tgt_vocab_size)
                outputs, src_cls_f, tgt_cls_f, gate_mask = model_txt(src, src_att, tgt, tgt_att, simg, inter_ids)

                loss_ce = criterion(outputs.view(-1, outputs.shape[-1]), tgt.view(-1))
                loss_sim = sim_loss(src_cls_f, tgt_cls_f)
                loss_bce = bce_loss(gate_mask, inter_ids).masked_select(src.ne(tokenizer.pad_token_id)).sum() / src_att.sum()
                # print(gate_mask)
                hard_gate_mask = (F.sigmoid(gate_mask)>config.theta).float()
                # print(hard_gate_mask)
                num_correct += (hard_gate_mask==inter_ids).masked_select(src.ne(tokenizer.pad_token_id)).sum()
                num_total += src_att.sum()
                # print("train: loss_ce({}), loss_mi({})".format(loss_ce_1.item(), loss_mi.item()))

                return_dict['loss_ce'] = loss_ce
                return_dict['loss_bce'] = loss_bce
                return_dict['loss_sim'] = loss_sim
                return_dict['t_loss'] = loss_ce + config.sim_alpha*loss_sim + config.ce2_beta*loss_bce
            scaler.scale(return_dict['t_loss']).backward()
            # nn.utils.clip_grad_norm_(model_txt.parameters(), max_norm=5, norm_type=2)
            scaler.step(optimz_txt)
            scaler.update()
            # optimz_txt.zero_grad()
            # outputs_1 = model_txt(src, src_att, tgt, tgt_att, simg)
            # loss_ce_1 = criterion(outputs_1.view(-1, outputs_1.shape[-1]), tgt.view(-1))
            # return_dict['loss_ce_1'] = loss_ce_1
            # return_dict['t_loss'] = loss_ce_1
            # return_dict['t_loss'].backward()
            # optimz_txt.step()

            params["updates"] += 1
            for key in return_dict:
                log_vars[key] += return_dict[key].item()


            # if eval_updates:
            if params["updates"]%eval_updates==0:
                ## report
                print("train loss: ", end='')
                for key in log_vars:
                    print("{}: {}, ".format(key, log_vars[key]/eval_updates), end="")
                    log_vars[key] = 0
                print("Acc: {:.3f}".format(num_correct / num_total))
                num_correct = num_total = 0
                
                print("train cost: {}s".format(time.time()-params['report_time']))
                params['report_time'] = time.time()
                ## evaluation
                print("evaluating after %d updates...\r" % params["updates"])
                print("--valid--")
                # print(model_txt.prompt_learner()[0])
                score = eval_model(model_txt, data, config, params)
                print("valid cost: {}s".format(time.time()-params['report_time']))
                params['report_time'] = time.time()
                
                if score <= params['metric'][-1]:
                    patience += 1
                    print('hit patience %d' % patience)
                if patience >= 5:
                    num_trial += 1
                    print('hit #%d trial' % num_trial)
                    if num_trial >= LIMIT_NUM:
                        print('early stop!')
                        print('the best model is from iteration [%d]' % best_model_iter,)
                        return
                    # decay lr, and restore from previously best checkpoint
                    lr = optimz_txt.param_groups[0]['lr'] * 0.5
                    print('load previously best model %s and decay learning rate to %f' % (params['best_model_path'], lr))
                    # load model
                    checkpoints = torch.load(params['best_model_path'], map_location=lambda storage, loc: storage)
                    model_txt.load_state_dict(checkpoints['model_txt'])
                    model_txt = model_txt.to(device)
                    print('restore parameters of the optimizers')
                    optimz_txt.load_state_dict(checkpoints["optim_txt"])
                    # set new lr
                    for param_group in optimz_txt.param_groups:
                        param_group['lr'] = lr
                    # reset patience
                    patience = 0
                
                params['metric'].append(score)
                if score >= max(params['metric']):
                    # print("--test--")
                    # t_score = test_model(model_txt, data, config, params)
                    best_model_iter = params['updates']
                    params['best_model_path'] = params["log_path"] + "best_checkpoint.pt"
                    save_model(
                        params['best_model_path'],
                        model_txt,
                        optimz_txt,
                        params["updates"],
                        config,
                    )
                model_txt.train()

@torch.no_grad()
def test_model(model, data, config, params):
    device = config.device
    model.to(device)
    model.eval()
    reference, candidate, source = [], [], []
    test_loader = data['test_loader']
    num_correct, num_total = 0, 0
    inter_str, ori_inters = [], []
    
    for ori_src, ori_tgt, img, ori_inter in test_loader:
        img = utils.token_img(img)
        src, src_att, tgt, tgt_att = utils.token_seq(ori_src, ori_tgt, tokenizer, eval=False)
        inter_ids = utils.get_intersection_idsmask(src, tgt)
        src = src.to(device)
        src_att = src_att.to(device)
        img = img.to(device)
        inter_ids = inter_ids.to(device)

        with torch.no_grad():
            samples, gate_mask = model.generate_text(
                src, src_att, img, inter_ids, beam_size=10,
            )
        hard_gate_mask = (F.sigmoid(gate_mask)>config.theta).float()
        # print(hard_gate_mask)
        num_correct += (hard_gate_mask==inter_ids).masked_select(src.ne(tokenizer.pad_token_id)).sum()
        num_total += src_att.sum()
        inter_str += [tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True)\
             for w in utils.del_tensor_0_nodim(hard_gate_mask*src, max_len=50, pad_token_id=tokenizer.pad_token_id)]

        candidate += samples
        source += ori_src
        reference += ori_tgt
        ori_inters += ori_inter
    
    with open(params['log_path']+"test_reference.txt", 'w') as f:
        for i in reference:
            f.write(str(i)+'\n')
    with open(params['log_path']+"test_candidate.txt", 'w') as f:
        for i in candidate:
            f.write(str(i)+'\n')
    with open(params['log_path']+"test_candidate_inter.txt", 'w') as f:
        for i in inter_str:
            f.write(str(i)+'\n')


    score = utils.rouge(
        reference, candidate, print
    )
    print("test Acc: {:.3f}".format(num_correct / num_total))
    for i, s_a in enumerate(inter_str):
        if len(s_a)<=0:
            inter_str[i]="<empty>"
    inter_score = utils.rouge(
        ori_inters, inter_str, print
    )
    return score

@torch.no_grad()
def eval_model(model_txt, data, config, params):
    device = config.device
    model_txt.to(device)
    model_txt.eval()
    reference, candidate, source = [], [], []
    valid_loader = data["valid_loader"]
    num_correct, num_total = 0, 0
    inter_str, ori_inters = [], []

    for ori_src, ori_tgt, img, ori_inter in valid_loader:
        img = utils.token_img(img)
        src, src_att, tgt, tgt_att = utils.token_seq(ori_src, ori_tgt, tokenizer, eval=False)
        inter_ids = utils.get_intersection_idsmask(src, tgt)
        src = src.to(device)
        src_att = src_att.to(device)
        img = img.to(device)
        inter_ids = inter_ids.to(device)

        with torch.no_grad():
            samples, gate_mask = model_txt.generate_text(
                src, src_att, img, inter_ids, beam_size=1,
            )
            # print(len(samples))
        hard_gate_mask = (F.sigmoid(gate_mask)>config.theta).float()
        # print(hard_gate_mask)
        num_correct += (hard_gate_mask==inter_ids).masked_select(src.ne(tokenizer.pad_token_id)).sum()
        num_total += src_att.sum()
        inter_str += [tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True)\
             for w in utils.del_tensor_0_nodim(hard_gate_mask*src, max_len=50, pad_token_id=tokenizer.pad_token_id)]
        
        candidate += samples
        source += ori_src
        reference += ori_tgt
        ori_inters += ori_inter
    
    with open(params['log_path']+"valid_reference.txt", 'w') as f:
        for i in reference:
            f.write(str(i)+'\n')
    with open(params['log_path']+"valid_candidate_{}.txt".format(params['updates']), 'w') as f:
        for i in candidate:
            f.write(str(i)+'\n')
    with open(params['log_path']+"valid_candidate_inter_{}.txt".format(params['updates']), 'w') as f:
        for i in inter_str:
            f.write(str(i)+'\n')

    # print(len(reference), len(candidate))
    score = utils.rouge(
        reference, candidate, print
    )
    print("valid Acc: {:.3f}".format(num_correct / num_total))
    for i, s_a in enumerate(inter_str):
        if len(s_a)<=0:
            inter_str[i]="<empty>"
    inter_score = utils.rouge(
        ori_inters, inter_str, print
    )
    return score




# save model
def save_model(path, model_txt, optim_txt, updates, config):
    model_txt_state_dict = model_txt.state_dict()
    optim_txt_state_dict = optim_txt.state_dict()
    checkpoints = {
        "model_txt": model_txt_state_dict,
        "config": config,
        "updates": updates,
        "optim_txt": optim_txt_state_dict,
    }
    torch.save(checkpoints, path)


def main():
    time1_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')
    print("start: ", time1_str)
    config = utils.get_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)
    utils.set_seed(config.seed)
    config.device = torch.device("cuda:{}".format(config.gpu))
    # config.device = torch.device("{}".format("cpu"))

    st_time = time.time()
    checkpoints = None
    model_txt, optimizer_txt  = build_model(checkpoints, config)
    print("cost {}s\n".format(time.time()-st_time))

    st_time = time.time()
    data = load_datav2(config)
    print("cost {}s\n".format(time.time()-st_time))
    print(f"len(train_loader):{len(data['train_loader'])}")
    print(f"len(valid_loader):{len(data['valid_loader'])}")

    params = {
        "updates": 0,
        "report_total": 0,
        "report_time": time.time(),
        "log_path": os.path.join(config.logdir, config.prename) + "/",
        "best_model_path": "",
    }
    os.makedirs(params['log_path'], exist_ok=True)

    params['metric'] = [0]

    scaler = GradScaler()
    if config.mode == "train":
        print("step four:\n")
        train_model(model_txt, data, optimizer_txt, params, config, scaler)

        print("Best %s score: %.3f\n" % ("Rouge", max(params['metric'])))
        model_path = params["log_path"] + "best_checkpoint.pt"
        checkpoints = torch.load(model_path, map_location=lambda storage, loc: storage)
        model_txt.load_state_dict(checkpoints['model_txt'])
        model_txt = model_txt.to(config.device)
        score = test_model(model_txt, data, config, params)
        print(f"Best score: {score}")       
    else:
        model_path = params["log_path"] + "best_checkpoint.pt"
        checkpoints = torch.load(model_path, map_location=lambda storage, loc: storage)
        del checkpoints['model_txt']['prompt_learner.token_prefix'], checkpoints['model_txt']['prompt_learner.token_suffix']
        model_txt.load_state_dict(checkpoints['model_txt'])
        model_txt = model_txt.to(config.device)
        score = test_model(model_txt, data, config, params)
        print(f"Best score: {score}") 

    time2_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')
    print("over: ", time2_str)


if __name__=='__main__':
    # utils.wait_unless(59276)
    print("\n\n------------------\n\n")
    main()
    print("\n\n------------------\n\n")


# python -u trainv20.py --batch_size 16 --sim_alpha 0.1 --ce2_beta 0.5 --theta 0.1 --n_prompt 8 --gpu 0  --prename v20
