# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

from EncoderRNN import EncoderRNN

class Lang:
    def __init__(self, vocab):
        self.index2word = {}
        self.word2index = {}
        for i in range(len(vocab)):
            self.index2word[i] = vocab[i]
            self.word2index[vocab[i]] = i      

    def indexFromSentence(self, sentence, flag_list=True):
        list_ = sentence if flag_list else sentence.lower().split()
        list_idx = []
        for word in list_:
            list_idx.append(self.word2index[word] if word in self.word2index else self.word2index['<unk>'])
        return list_idx

    def VariablesFromSentences(self, sentences, flag_list=True, use_cuda=True):
        '''
        if sentence is a list of word, flag_list should be True in the training 
        '''
        indexes = [self.indexFromSentence(sen, flag_list) for sen in sentences]
        inputs = Variable(torch.LongTensor(indexes))
        return inputs.cuda() if use_cuda else inputs

class rnnCapsule(object):
    def __init__(self,
            dim_input,
            dim_hidden,
            n_layers,
            n_label,
            batch_size,
            max_length,
            learning_rate,
            lr_word_vector=0.01,
            weight_decay=0,
            vocab=None,
            embed=None,
            embed_dropout_rate=0.,
            cell_dropout_rate=0.,
            final_dropout_rate=0.,
            bidirectional=True,
            optim_type="Adam",
            rnn_type="LSTM",
            use_cuda=True):
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.lang = Lang(vocab)
        self.model = EncoderRNN(dim_input, dim_hidden, n_layers, n_label, len(vocab), 
                embed_dropout_rate, cell_dropout_rate, final_dropout_rate, embed, bidirectional, rnn_type, use_cuda)
        if self.use_cuda:
            self.model.cuda()
        self.optimizer = getattr(optim, optim_type)([
                                        {'params': self.model.base_params},
                                        {'params': self.model.embed.parameters(), 'lr': lr_word_vector, 'weight_decay': 0}, 
                                    ], lr=self.learning_rate, weight_decay=weight_decay)
        self.encoder_hidden = self.model.init_hidden(self.batch_size)

    def get_batch_data(self, batched_data):
        input_sentence, tensor_label, sen_len = batched_data['sentence'], batched_data['labels'], batched_data['sentence_length']
        input_variable = self.lang.VariablesFromSentences(input_sentence, True, self.use_cuda)
        tensor_label = Variable(torch.from_numpy(batched_data['labels']))
        tensor_label = tensor_label.cuda() if self.use_cuda else tensor_label
        return input_variable, tensor_label, sen_len

    def stepTrain(self, batched_data, inference=False):
        # Turn on training mode which enables dropout.
        self.model.eval() if inference else self.model.train()
        input_variable, tensor_label, sen_len = self.get_batch_data(batched_data)
        hidden = self.model.init_hidden(len(batched_data['sentence_length']))
        
        if inference == False:
            # zero the parameter gradients
            self.optimizer.zero_grad()

        loss_sim, prob = self.model(input_variable, hidden, sen_len, tensor_label)
        loss_hinge_classify = F.multi_margin_loss(prob, tensor_label)
        loss_hinge = F.multi_margin_loss(loss_sim, tensor_label)
        loss = loss_hinge_classify + loss_hinge

        if inference == False:
            loss.backward()
            self.optimizer.step()
        
        return np.array([loss.data.cpu().numpy(), loss_hinge_classify.data.cpu().numpy(), loss_hinge.data.cpu().numpy()]).reshape(3), prob.data.cpu().numpy()

    def save_model(self, dir, idx):
        os.mkdir(dir) if not os.path.isdir(dir) else None
        torch.save(self, '%s/model%s.pkl' % (dir, idx))