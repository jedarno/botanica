import numpy as np
import torch
from torch import nn
from torch.functional import norm
from torch.nn import init

class LPI(nn.Module):
    """
    Local Patch Interaction module that allows explicit communication between tokens in 3x3 windows to augment the
    implicit communication performed by the block diagonal scatter attention. Implemented using 2 layers of separable
    3x3 convolutions with GeLU and BatchNorm2d
    From: https://blog.ceshine.net/post/xcit-part-2/ 
    referecing: https://github.com/rwightman/pytorch-image-models/blob/763329f23f675626e657f012e633fca5ea0985ed/timm/models/xcit.py#L180
    """

    def __init__(self, in_features, out_features=None, act_layer=nn.GELU, kernel_size=3):
        super().__init__()
        out_features = out_features or in_features

        padding = kernel_size // 2

        self.conv1 = torch.nn.Conv2d(
            in_features, in_features, kernel_size=kernel_size, padding=padding, groups=in_features)
        self.act = act_layer()
        self.bn = nn.BatchNorm2d(in_features)
        self.conv2 = torch.nn.Conv2d(
            in_features, out_features, kernel_size=kernel_size, padding=padding, groups=out_features)

    def forward(self, x, H: int, W: int):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)
        return x


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

class UfoClassHead(nn.Module):
    """
    !!! NO CONV for now !!!
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
        super(UfoClassHead, self).__init__()
        self.UFO_att = UFOAttention(hidden_layers, d_k, d_v, num_heads, dropout=0.1)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=768, out_features=3072, bias=True),
            nn.GELU(approximate='none'),
            nn.Linear(in_features=3072, out_features=768, bias=True)
                )

        #Encoder Block Layer Norms 
        self.ln1 = nn.LayerNorm((768,), elementwise_affine = True)
        self.ln2 = nn.LayerNorm((768,), elementwise_affine = True)

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
    def forward(self, x):

        #UFO_Encoder_block
        x = self.ln1(x)
        x = self.UFO_att(x, x, x)
        x = self.ln2(x)
        x = self.mlp(x)
        #Normalising at end of encoder
        x = self.ln(x)
        x = x[:, 0]
        x = self.linear_out(x)
        return x


