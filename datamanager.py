# -*- coding: utf-8 -*-
import numpy as np

class DataManager(object):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

    def load_data(self, path, fname):
        with open('%s/%s' % (path, fname)) as f:
            lines = [line.strip() for line in f.readlines()]
        data = []
        for line in lines:
            dict_tmp = eval(line)
            dict_tmp['sentence'] = dict_tmp['sentence'].lower().split()
            data.append(dict_tmp)
        return data

    def build_vocab(self, path, data):
        print("Creating vocabulary...")
        vocab = {}
        for pair in data:
            for token in pair['sentence']:
                if token in vocab:
                    vocab[token] += 1
                else:
                    vocab[token] = 1
        vocab_list =  sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > self.FLAGS.voc_size:
            vocab_list = vocab_list[:self.FLAGS.voc_size]
        vocab_list.append('<unk>')

        print("Loading word vectors...")
        vectors = {}
        with open('%s' % path) as f:
            for line in f:
                s = line.strip()
                word = s[:s.find(' ')]
                vector = s[s.find(' ')+1:]
                vectors[word] = vector
        
        embed = []
        num_not_found, num_found = 0, 0
        for word in vocab_list:
            if word in vectors:
                vector = map(float, vectors[word].split())
                num_found = num_found + 1
            else:
                num_not_found = num_not_found + 1
                vector = np.zeros((self.FLAGS.word_dim), dtype=np.float32)
            embed.append(vector)
        print('%s words found in vocab' % num_found)
        print('%s words not found in vocab' % num_not_found)
        embed = np.array(embed, dtype=np.float32)
        return vocab_list, embed, vocab

    def gen_batched_data(self, data, flag_label_respresentation=2):
        '''
        flag_label_respresentation
        0, scalar output
        1, vector output, negative idx is 0, for cross entropy
        2, vector output, negative idx is -1, for hinge margin loss
        '''
        max_len_ = max([len(item['sentence']) for item in data])
        max_len = self.FLAGS.max_length if max_len_ > self.FLAGS.max_length else max_len_
        sentence, sentence_length, labels = [], [], []
        def padding(sent, l):
            return sent + ['_PAD'] * (l-len(sent))

        def scalar2vect(num, n_label):
            if flag_label_respresentation == 0:
                return num
            vect_re = [-1] *n_label if flag_label_respresentation == 2 else [0] * n_label
            vect_re[num] = 1
            return vect_re
            
        for item in data:
            if len(item['sentence']) < 1:
                print(item)
                exit()
            if len(item['sentence']) > max_len:
                sentence.append(item['sentence'][:max_len])
                sentence_length.append(max_len)
                labels.append(scalar2vect(item['label'], self.FLAGS.n_label))
            else:
                sentence.append(padding(item['sentence'], max_len))
                sentence_length.append(len(item['sentence']))
                labels.append(scalar2vect(item['label'], self.FLAGS.n_label))

        # sort by the length of sentence
        idx = np.argsort(sentence_length)[::-1]
        sentence = np.array(sentence)[idx]
        labels = np.array(labels)[idx]
        sentence_length = np.array(sentence_length)[idx]

        batched_data = {'sentence': sentence, 'labels': labels,
                'sentence_length': sentence_length}
        return batched_data

    def word2vec_pre_select(self, mdict, word2vec_file_path, save_vec_file_path):
        list_seledted = []
        with open(word2vec_file_path) as f:
            for line in f:
                tmp = line.strip().split(' ', 1)
                if mdict.has_key(tmp[0]):
                    list_seledted.append(line.strip())
        open(save_vec_file_path, 'w').write('\n'.join(list_seledted))