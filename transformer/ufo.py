import numpy as np
import torch
from torch import nn
from torch.functional import norm
from torch.nn import init


def XNorm(x,gamma):
    norm_tensor=torch.norm(x,2,-1,True)
    return x*gamma/norm_tensor


class UFOAttention(nn.Module):
    '''
    Scaled dot-product attention
    from: https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/UFOAttention.py
    '''

    def __init__(self, d_model, d_k, d_v, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(UFOAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)
        self.gamma=nn.Parameter(torch.randn((1,h,1,1)))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        kv=torch.matmul(k, v) #bs,h,c,c
        kv_norm=XNorm(kv,self.gamma) #bs,h,c,c
        q_norm=XNorm(q,self.gamma) #bs,h,n,c
        out=torch.matmul(q_norm,kv_norm).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        
        return out

class UFO_Class_Head(nn.module):
"""
A block that implements a transformer block taht uses UFO attention with a convolutional layer. Should be implementable
as a ViT encoder block, using multihead UFO attention instead of multihead self attention. 
"""

    def __init__(self, hidden_layers, d_k, d_v, num_heads, n_classes,  dropout=None, model='a'):
        """
        In the constructor I want to:
        (1) Call constructor of superclass
        (2) Call instance of UFO_Att
        (3) Three model set ups to be used:
            (3a) attention on fmap(inc cls tokens)
            (3b) Q represents cls tokens
            (3c) attention applied after seperating class tokens (class attention)
        """
        self.UFO_att = UFOAttention(hidden_layers, d_k, d_v, num_heads, dropout=None)
        self.same_conv = nn.conv2d((197, hidden_layers), (197, hidden_layers), kernel_size=(3,3), stride=(3,3))  
        self.mlp = nn.Sequential(
            nn.Linear(in_features=768, out_features=3072, bias=True),
            nn.GELU(approximate='none'),
            nn.Linear(in_features=3072, out_features=768, bias=True)
                )
        #(3a) This should be in forward and not use Sequential
        """
        if model = 'a':
            self.ufo_encoder = nn.Sequential(
                nn.LayerNorm((768,), elementwise_affine = True)
                self.UFO_att
                self.LayerNorm((768,), elementwise_affine = True)
                self.same_conv,
                self.LayerNorm((768,), elementwise_affine = True)
                self.mlp
                    )
        """
    def forward(x):
        return self.classifier(x)

