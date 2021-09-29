import numpy as np
import pickle
import random
import os, re

import torch
from collections import Counter, defaultdict

from .helper import STR2DICT, SWAP_k_V
from .helper import english_tag2idx, tokenize

class Treebank(object):
    def __init__(self, data_file, require_tagset=False):
        data = pickle.load(open(data_file, 'rb')) #get text data
        self.sents = self._long_tensor(data['source'])
        self.other_data = data['other_data']
        self.sent_lengths = self._long_tensor(data['source_l'])
        self.batch_size = self._long_tensor(data['batch_l'])
        self.batch_idx = self._long_tensor(data['batch_idx'])
        self.vocab_size = data['vocab_size'][0]
        self.num_batches = self.batch_idx.size(0)
        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']

        self.tag2idx, self.idx2tag = self._build_tagset(require_tagset)
        self._tokenizer = None
    
    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer):
        self._tokenizer = tokenizer

    def expand_vocab(self, root, name):
        ifile = f"{root}/{name}"
        if not os.path.isfile(ifile):
            return False

        def read_word_list(ifile):
            vocab = Counter()
            with open(ifile, 'r') as fr:
                for line in fr:
                    word, cnt = line.strip().split()
                    vocab[word] = int(cnt)
            return vocab
        
        k = 5 # the minimum frequency
        counter = read_word_list(ifile)
        sub_keys = [t for t, v in counter.most_common() if v >= k]

        idx2word_new = dict()
        for word in sub_keys:
            if self.word2idx.get(word, None) is None:
                idx = len(self.word2idx)
                self.word2idx[word] = idx 
                idx2word_new[idx] = word 
        self.idx2word.update(idx2word_new)
        assert len(self.word2idx) == len(self.idx2word)
        self.vocab_size = len(self.word2idx)
        return True

    def sync_vocab(self, data):
        self.word2idx = data.word2idx
        self.idx2word = data.idx2word
        self.vocab_size = data.vocab_size

    def _long_tensor(self, x):
        return torch.from_numpy(np.asarray(x)).long()

    def __len__(self):
        return self.num_batches

    def _build_tagset(self, build):
        if not build:
            return None, None
        PAD, UNK, BOS, EOS = ("<pad>","<unk>","<s>","</s>")
        tag2idx = {PAD: 0, UNK: 1, BOS: 2, EOS: 3}
        for sample in self.other_data:
            for tag in sample[1]: 
                if tag not in tag2idx:
                    tag2idx[tag] = len(tag2idx)     
        idx2tag = {}
        for k, v in tag2idx.items():
            idx2tag[v] = k
        return tag2idx, idx2tag

    def _encode_subwords(self, items):
        if self.tokenizer is None:
            return items
        indexes, targets = list(), list()
        idx_lengths, mul_lengths = list(), list()
        for item in items:
            if len(item) == 7:
                sentence = item[0]
                _, mul_target, mul_index = tokenize(
                    sentence, self._tokenizer, 0, float("inf"), lower_case=True
                )
                item.extend([mul_target, mul_index])
            mul_lengths.append(len(item[-2]))
            targets.append(item[-2])
            idx_lengths.append(len(item[-1]))
            indexes.append(item[-1])

        mul_max_len = max(mul_lengths)
        mul_targets = torch.zeros(len(items), mul_max_len).fill_(
            self.tokenizer.pad_token_id
        ).long()
        for i, length in enumerate(mul_lengths):
            mul_targets[i, : length] = torch.tensor(
                self.tokenizer.convert_tokens_to_ids(targets[i])
            )

        idx_max_len = max(idx_lengths)
        mul_indexes = torch.zeros(len(items), idx_max_len).fill_(-100).long()
        for i, length in enumerate(idx_lengths):
            mul_indexes[i, : length] = torch.tensor(indexes[i][: length])
        items = (items,) + (mul_targets, mul_indexes)
        return items 

    def _encode_tags(self, samples, length):
        if self.tag2idx is None:
            return None
        tag_idxes = torch.tensor([
            [self.tag2idx[tag] for tag in sample] for sample in samples
        ])
        assert tag_idxes.size(-1) == length - 2
        return tag_idxes

    def __getitem__(self, idx):
        assert(idx <= self.num_batches and idx >= 0), f"{idx} not in [0, {self.num_batches}]"
        start_idx = self.batch_idx[idx]
        end_idx = start_idx + self.batch_size[idx]
        length = self.sent_lengths[idx].item()
        sents = self.sents[start_idx:end_idx]
        other_data = self.other_data[start_idx:end_idx]
        sent_str = [d[0] for d in other_data]
        tags = [d[1] for d in other_data]
        actions = [d[2] for d in other_data]
        binary_tree = [d[3] for d in other_data]
        spans = [d[5] for d in other_data]
        batch_size = self.batch_size[idx].item()
        # original data includes </s>, which we don't need
        data_batch = [sents[:, 1:length-1], length-2, batch_size, actions, 
                      spans, binary_tree, other_data]
        # additional encodings 
        data_batch[-1] = self._encode_subwords(other_data)
        data_batch.insert(0, self._encode_tags(tags, length))
        return data_batch

def build_treebank(cfg, echo, data_name, train=False, key=None):
    data_file = f"{cfg.data_root}/" + data_name
    if key is not None:
        data_file = data_file.format(key)
    data = Treebank(data_file)
    if train and os.path.isfile(f"{cfg.pair_root}/{cfg.pair_name}"):
        old_size = data.vocab_size
        vocab_name = cfg.extra_vocab
        if "english" in data_file:
            vocab_name += ".en"
        elif "chinese" in data_file:
            vocab_name += ".zh"
        flag = data.expand_vocab(cfg.pair_root, vocab_name)
        if flag: # show expanded vocab
            new_size = data.vocab_size
            echo(f"Vocab has been expanded - old: {old_size} vs new: {new_size}")
    if "english" in data_file:
        data.tag2idx = STR2DICT(english_tag2idx)
        data.idx2tag = SWAP_k_V(data.tag2idx) 
    return data
