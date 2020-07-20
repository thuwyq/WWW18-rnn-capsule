# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import time, random, argparse
import numpy as np
from sklearn.metrics import confusion_matrix
from datamanager import DataManager
from rnn_capsule import rnnCapsule
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--voc_size', type=int, default=32768)
parser.add_argument('--word_dim', type=int, default=300, choices=[100, 300])
parser.add_argument('--hidden_dim', type=int, default=256, choices=[128, 256, 512])
parser.add_argument('--n_layer', type=int, default=2)
parser.add_argument('--n_label', type=int, default=2, choices=[2, 3, 5])
parser.add_argument('--bidirectional', type=bool, default=False)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--lr_word_vector', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--embed_dropout', type=float, default=0.3)
parser.add_argument('--cell_dropout', type=float, default=0.5)
parser.add_argument('--final_dropout', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--max_length', type=int, default=64)
parser.add_argument('--iter_num', type=int, default=32*256)
parser.add_argument('--per_checkpoint', type=int, default=32)
parser.add_argument('--seed', type=int, default=1705216)
parser.add_argument('--rnn_type', type=str, default="LSTM", choices=["LSTM", "GRU"])
parser.add_argument('--optim_type', type=str, default="Adam", choices=["Adam", "Adadelta", "RMSprop", "Adagrad"])
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--wordvec_name', type=str, default='glove.840B.300d.txt')
parser.add_argument('--name_model', type=str, default='RNN-Capsule-master')
FLAGS = parser.parse_args()
print(FLAGS)

np.random.seed(FLAGS.seed)
random.seed(FLAGS.seed)
torch.manual_seed(FLAGS.seed)
torch.backends.cudnn.enabled = False
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed_all(FLAGS.seed)

def train(model, datamanager, data_train):
    selected_data = [random.choice(data_train) for i in range(FLAGS.batch_size)]
    batched_data = datamanager.gen_batched_data(selected_data, flag_label_respresentation=0)
    loss, _ = model.stepTrain(batched_data)
    return loss

def evaluate(model, datamanager, data_, name=None):
    loss = np.zeros((3, ))
    st, ed, times = 0, FLAGS.batch_size, 0
    pred_matrix = []
    y_true = []
    while st < len(data_):
        selected_data = data_[st:ed]
        batched_data = datamanager.gen_batched_data(selected_data, flag_label_respresentation=0)
        outputs, pred_ = model.stepTrain(batched_data, inference=True)
        pred_matrix.extend(pred_)
        y_true.extend(batched_data['labels'])
        loss += outputs
        st, ed = ed, ed+FLAGS.batch_size
        times += 1
    pred_vector = np.argmax(np.array(pred_matrix), axis=1)
    c_m = confusion_matrix(np.array(y_true), pred_vector, labels=range(FLAGS.n_label))
    loss /= times
    accuracy = np.sum([c_m[i][i] for i in xrange(FLAGS.n_label)]) / np.sum(c_m)
    return loss, accuracy, c_m

if __name__ == "__main__":
    dataset_name = ['train', 'dev', 'test']
    datamanager = DataManager(FLAGS)
    data = {}
    for tmp in dataset_name:
        data[tmp] = datamanager.load_data(FLAGS.data_dir, '%s.json' % tmp)
    vocab, embed, vocab_dict = datamanager.build_vocab('%s/%s' % (FLAGS.data_dir, FLAGS.wordvec_name), data['train'])

    print('model parameters: %s' % str(FLAGS))
    print("Use cuda: %s" % use_cuda)
    print('train data: %s, dev data: %s, test data: %s' % (len(data['train']), len(data['dev']), len(data['test'])))

    model = rnnCapsule(
            FLAGS.word_dim,
            FLAGS.hidden_dim, 
            FLAGS.n_layer,
            FLAGS.n_label,
            batch_size=FLAGS.batch_size,
            max_length=FLAGS.max_length,
            learning_rate=FLAGS.learning_rate,
            lr_word_vector=FLAGS.lr_word_vector,
            weight_decay=FLAGS.weight_decay,
            vocab=vocab,
            embed=embed,
            embed_dropout_rate=FLAGS.embed_dropout,
            cell_dropout_rate=FLAGS.cell_dropout,
            final_dropout_rate=FLAGS.final_dropout,
            bidirectional=FLAGS.bidirectional,
            optim_type=FLAGS.optim_type,
            rnn_type=FLAGS.rnn_type,
            use_cuda=use_cuda)

    loss_step, time_step = np.zeros((3,)), 1e10

    start_time = time.time()    
    for step in range(FLAGS.iter_num):
        if step % FLAGS.per_checkpoint == 0:
            show = lambda a: '[%s]' % (' '.join(['%.2f' % x for x in a]))
            time_step = time.time() - start_time
            print("------------------------------------------------------------------")
            print('Time of iter training %.2f s' % time_step)
            print("On iter step %s:, global step %d learning rate %.4f Loss-step %s"
                    % (step/FLAGS.per_checkpoint, step, model.optimizer.param_groups[0]['lr'], show(np.exp(loss_step))))
            # model.save_model("%s/%s" %("./model", FLAGS.name_model), int(step/FLAGS.per_checkpoint))

            for name in dataset_name:
                loss, acc, c_m = evaluate(model, datamanager, data[name], name)
                print('In dataset %s: Loss is %s, Accuracy is %s' % (name, show(np.exp(loss)), acc))
                print('\n%s' % c_m)
            
            start_time = time.time()
            loss_step = np.zeros((3, ))

        loss_step += train(model, datamanager, data['train']) / FLAGS.per_checkpoint
