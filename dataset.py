import os

import numpy as np
import torch
from tqdm import tqdm

import utils


def load_data(config):
    """
    load data.
    update "data" due to the saved path in the pickle file
    :return: a dict with data and vocabulary
    """
    print("loading data...\n")
    

    train_set = utils.BiDataset(os.path.join(config.data_path, "train"), os.path.join(config.data_path, "images_train"), True)
    test_set = utils.BiDataset(os.path.join(config.data_path, "test"), os.path.join(config.data_path, "images_test"), False)
    valid_set = utils.BiDataset(os.path.join(config.data_path, "valid"), os.path.join(config.data_path, "images_valid"), False)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True,num_workers=config.num_workers, drop_last=True, collate_fn=utils.collate_function)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=utils.collate_function)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=utils.collate_function)
    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "valid_loader": valid_loader,
    }


def load_datav2(config):
    """
    load datav2.
    update "data" due to the saved path in the pickle file
    :return: a dict with data and vocabulary
    """
    print("loading data...\n")
    

    train_set = utils.BiDatasetv2(os.path.join(config.data_path, "train"), os.path.join(config.data_path, "images_train"), True)
    test_set = utils.BiDatasetv2(os.path.join(config.data_path, "test"), os.path.join(config.data_path, "images_test"), False)
    valid_set = utils.BiDatasetv2(os.path.join(config.data_path, "valid"), os.path.join(config.data_path, "images_valid"), False)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True,num_workers=config.num_workers, drop_last=True, collate_fn=utils.collate_functionv2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=utils.collate_functionv2)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=utils.collate_functionv2)
    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "valid_loader": valid_loader,
    }

def load_datav3(config):
    """
    load datav3.
    update "data" due to the saved path in the pickle file
    :return: a dict with data and vocabulary
    """
    print("loading data...\n")
    

    train_set = utils.BiDatasetv3(os.path.join(config.data_path, "train"), os.path.join(config.data_path, "images_train"), True)
    test_set = utils.BiDatasetv3(os.path.join(config.data_path, "test"), os.path.join(config.data_path, "images_test"), False)
    valid_set = utils.BiDatasetv3(os.path.join(config.data_path, "valid"), os.path.join(config.data_path, "images_valid"), False)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True,num_workers=config.num_workers, drop_last=True, collate_fn=utils.collate_functionv2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=utils.collate_functionv2)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=utils.collate_functionv2)
    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "valid_loader": valid_loader,
    }

def load_data_txt(config):
    """
    load data.
    update "data" due to the saved path in the pickle file
    :return: a dict with data and vocabulary
    """
    print("loading data...\n")
    

    train_set = utils.BiDataset_t(os.path.join(config.data_path, "train"))
    test_set = utils.BiDataset_t(os.path.join(config.data_path, "test"))
    valid_set = utils.BiDataset_t(os.path.join(config.data_path, "valid"))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True,num_workers=config.num_workers, drop_last=True, collate_fn=utils.collate_function_t)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=utils.collate_function_t)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=utils.collate_function_t)
    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "valid_loader": valid_loader,
    }

def load_data_txtv2(config):
    """
    load data.
    update "data" due to the saved path in the pickle file
    :return: a dict with data and vocabulary
    """
    print("loading data...\n")
    

    train_set = utils.BiDataset_tv2(os.path.join(config.data_path, "train"))
    test_set = utils.BiDataset_tv2(os.path.join(config.data_path, "test"))
    valid_set = utils.BiDataset_tv2(os.path.join(config.data_path, "valid"))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True,num_workers=config.num_workers, drop_last=True, collate_fn=utils.collate_function_t)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=utils.collate_function_t)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=utils.collate_function_t)
    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "valid_loader": valid_loader,
    }




if __name__ == "__main__":
    from transformers import BartTokenizer
    print("dataset.py")
    config = utils.get_args()
    train_set = utils.BiDataset_t(os.path.join(config.data_path, "train"))
    valid_set = utils.BiDataset_t(os.path.join(config.data_path, "valid"))
    test_set = utils.BiDataset_t(os.path.join(config.data_path, "test"))
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True,num_workers=config.num_workers, drop_last=True, collate_fn=utils.collate_function_t)

#    len(train_loader):969
#    len(valid_loader):32
#    len(src_vocab):36915
#    len(tgt_vocab):26170
    src_lens = []
    tgt_lens = []
    inter_len = []
    cnt_0 = 0

    for ori_src, ori_tgt in tqdm(test_set):
        src_lens.append(len(ori_src.split()))
        tgt_lens.append(len(ori_tgt.split()))
        inter_str = utils.get_intersection(ori_src, ori_tgt)
        inter_len.append(len(inter_str.split()))

    print("src_mean: ", np.mean(src_lens))
    print("tgt_mean: ", np.mean(tgt_lens))
    print("cnt_0: ", cnt_0,":", cnt_0/len(src_lens)*100, "%")
    print("max src_len: ", max(src_lens))
    print("max tgt_len: ", max(tgt_lens))

    print("inter_mean: ", np.mean(inter_len))
    print("max inter: ", max(inter_len))

    
