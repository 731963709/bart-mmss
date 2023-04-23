import os
import random
import sys
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def set_cuda(config):
    use_cuda = config.use_cuda
    if use_cuda:
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True
    device = (
        torch.device("cuda:{}".format(config.gpu))
        if use_cuda
        else torch.device("cpu")
    )
    devices_id = config.gpu
    return device, devices_id


def set_tensorboard(summary_dir):
    # summary_dir = os.path.join(config.logdir, config.expname)
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    for file_name in os.listdir(summary_dir):
        if file_name.startswith("events.out.tfevents"):
            print(f"Event file {file_name} already exists")
            os.remove(os.path.join(summary_dir, file_name))
            print(f"Event file {file_name} removed")
    return SummaryWriter(summary_dir)


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def wait_unless(pid):
    import psutil
    while True:
        pl = psutil.pids()  #所有的进程列出来
        if pid in pl:
            time.sleep(15*60)
        else:
            break
    print("begin!")


# tensorboard --logdir=../experiments/only_mul/ --port 7007 --host 10.102.32.222
