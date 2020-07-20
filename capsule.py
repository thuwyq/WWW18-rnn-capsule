# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from attentionlayer import Attention

class Capsule(nn.Module):
    def __init__(self, dim_vector, final_dropout_rate, use_cuda=True):
        super(Capsule, self).__init__()
        self.dim_vector = dim_vector
        self.add_module('linear_prob', nn.Linear(dim_vector, 1))
        self.add_module('final_dropout', nn.Dropout(final_dropout_rate))
        self.add_module('attention_layer', Attention(attention_size=dim_vector, return_attention=True, use_cuda=use_cuda))

    def forward(self, vect_instance, matrix_hidden_pad, len_hidden_pad=None):
        r_s, attention = self.attention_layer(matrix_hidden_pad, torch.LongTensor(len_hidden_pad))
        prob = F.sigmoid(self.linear_prob(self.final_dropout(r_s)))
        r_s_prob = prob * r_s
        return prob, r_s_prob