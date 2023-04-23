from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm


class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False


class NT_Xent(nn.Module):
    def __init__(self, temperature=10.0):
        super(NT_Xent, self).__init__()
        self.temperature = temperature
        # self.temperature = nn.Parameter(torch.tensor(self.temperature, dtype=torch.float32))

        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        """
        z_i = F.normalize(z_i)
        z_j = F.normalize(z_j)
        B_size, dim_size = z_i.size()
        # print(B_size)
        N = B_size * 2
        # [2*B, D]
        out = torch.cat((z_i, z_j), dim=0)
        # print(out)
        # [2*B, 2*B]
        mm_out = torch.mm(out, out.t().contiguous())
        # print(mm_out)
        sim_matrix = torch.exp(mm_out / self.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(N, device=sim_matrix.device)).bool()

        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(N, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(z_i * z_j, dim=-1) / self.temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        return loss


class TextCnn(nn.Module):
    def __init__(self, hidden_size):
        super(TextCnn, self).__init__()
        filter_sizes = (2, 3, 4)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, hidden_size, (k, hidden_size)) for k in filter_sizes])
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size * len(filter_sizes), hidden_size)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = x.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

class L_Discriminator(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.cnn = TextCnn(hidden_size)
    def forward(self, local_f):
        ans = self.cnn(local_f) #[b,h]
        # print("local ans: ", ans.shape)
        return ans

class G_Discriminator(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.out = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(hidden_size*2, hidden_size, bias=False)),
                ('bn1', nn.BatchNorm1d(hidden_size)),
                ('relu1', nn.ReLU()),
                ('fc2', nn.Linear(hidden_size, hidden_size, bias=False)),
                ('bn2', BatchNorm1dNoBias(hidden_size)),
            ]))
    def forward(self, in_1, in_2):
        in_g = torch.cat([in_1, in_2], dim=-1)
        ans = self.out(in_g)
        # print("global ans: ", ans.shape)
        return ans



class DeepInfomax(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.discr_l = L_Discriminator(hidden_size*2)
        self.discr_g = G_Discriminator(hidden_size)

        self.maxpooling = nn.AdaptiveMaxPool1d(1)
    
    def change_z(self, z):
        batch_size, tgt_len, dim_size = z.shape
        z = z.transpose(1,2)
        z = self.maxpooling(z)
        z = z.transpose(1,2)
        return z.reshape((-1, dim_size))

    def forward(self, input_1:torch.Tensor, input_2:torch.Tensor):
        """
        input_1, input_2: [b,l,h]
        """
        input_1 = self.change_z(input_1)
        input_2 = self.change_z(input_2)

        input_1_prime = torch.cat([input_1[1:], input_1[0].unsqueeze(0)])

        # fusion = torch.cat([input_1, input_2], dim=-1)
        # fusion_prime = torch.cat([input_1_prime, input_2], dim=-1)

        # # local loss
        # Ej = -F.softplus(-self.discr_l(fusion)).mean()
        # Em = F.softplus(self.discr_l(fusion_prime)).mean()
        # LOCAL = (Em - Ej)

        Ej = -F.softplus(-self.discr_g(input_1, input_2)).mean()
        Em = F.softplus(self.discr_g(input_1_prime, input_2)).mean()
        GLOBAL = (Em - Ej)
        
        return GLOBAL*1.0
        return LOCAL*0.5 + GLOBAL*1.0


class LabelSmoothingLoss(nn.Module):

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = ignore_index
        super(LabelSmoothingLoss, self).__init__()
        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        output = F.log_softmax(output, dim=-1)
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class c(nn.Module):

    def __init__(self, hidden_dim=128, pretrained=True):
        super(ReLoss, self).__init__()
        self.logits_fcs = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.loss = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

        # load pretrained weights
        if pretrained:
            ckpt_url = 'https://github.com/hunto/ReLoss/releases/download/v1.0.0/reloss_cls.ckpt'
            print(f'Load checkpoint of ReLoss from url: {ckpt_url}')
            state_dict = torch.hub.load_state_dict_from_url(ckpt_url, map_location='cpu')
            self.load_state_dict(state_dict)

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        pos_probs = torch.gather(probs, 1, targets.unsqueeze(-1))
        hidden = self.logits_fcs(pos_probs)
        loss = self.loss(hidden).abs().mean()
        return loss


if __name__=="__main__":
    # m = nn.AdaptiveAvgPool1d(4)
    # input = torch.autograd.Variable(torch.randn(1, 64, 8))
    # output = m(input)
    # print(output.size())
    # exit(0)
    # textCnn = TextCnn(512)
    # in_a = torch.rand((16, 24, 512))
    # print(textCnn(in_a).shape)
    

    loss_fn = DeepInfomax(768)
    z_i, z_j = torch.rand((8, 380, 768)), torch.rand((8, 90, 768))
    print(loss_fn(z_i, z_j))

    # loss_fn = NT_Xent(1)
    # z_i, z_j = torch.rand((12, 380, 768)), torch.rand((12, 90, 768))
    # print(loss_fn(z_i, z_j))
