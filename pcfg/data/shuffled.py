import os
import math
import torch

import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence

from .treebank import Treebank 
from .helper import SortedBlockSampler, SortedRandomSampler, tokenize
from .helper import STR2DICT, SWAP_k_V
from .helper import english_tag2idx, tokenize

class ShuffledTreebank(Treebank, torch.utils.data.Dataset):
    """ Override Yoon's dataloaser and allow for varying lengths within a batch.
    """
    def __init__(
        self, data_file, require_tagset=False, batch_size=4, 
        min_length=1, max_length=100, main_col=1, nsample=float("inf")
    ):
        super(ShuffledTreebank, self).__init__(data_file)
        self._batch_size = batch_size # different from `batch_size` of the base class
        self.main_col = main_col
        self.dataset = list()
        for idx in range(self.num_batches):
            tags, sentences, length, batch_size, actions, gold_spans, gold_btrees, other_data = self._make_batch(idx)
            tags = [None] * batch_size if tags is None else tags
            sub_words, token_indice = ([None] * batch_size,) * 2
            if isinstance(other_data, tuple):
                sub_words, token_indice = other_data[1:]
                other_data = other_data[0]

            if length < min_length or length > max_length: 
                continue
            
            for b in range(batch_size):
                sample = (
                    tags[b], sentences[b], actions[b], gold_spans[b], gold_btrees[b], 
                    other_data[b], sub_words[b], token_indice[b]
                )
                self.dataset.append(sample) 

    def _make_batch(self, idx):
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

    def _shuffle(self):
        indice = torch.randperm(len(self)).tolist() 
        indice = sorted(indice, key=lambda k: len(self.dataset[k][self.main_col]))
        dataset = [self.dataset[k] for k in indice]
        self.dataset = dataset

    def __getitem__(self, index):
        tags, sentences, actions, gold_spans, gold_btrees, \
            other_data, sub_words, token_indice = self.dataset[index]
        sub_word_pad = None 
        if self.tokenizer is not None: # determined at running time
            sub_word_pad = self.tokenizer.pad_token_id 
            other_data = self._encode_subwords([other_data])
            mul_targets, mul_indexes = other_data[1:]
            mul_targets = mul_targets.squeeze(0)
            mul_indexes = mul_indexes.squeeze(0)
            sub_words, token_indice = mul_targets, mul_indexes
            other_data = other_data[0]
        word_pad = self.word2idx['<pad>']
        tag_pad = self.tag2idx['<pad>']
        return (
            tags, sentences, actions, gold_spans, gold_btrees, other_data, 
            sub_words, token_indice, tag_pad, word_pad, sub_word_pad
        )

    def __len__(self):
        return len(self.dataset)
    
def collate_fun(data):
    tags, sentences, actions, gold_spans, gold_btrees, other_data, \
        sub_words, token_indice, tag_pads, word_pads, sub_word_pads = list(zip(*data))
    lengths = torch.tensor([len(sentence) for sentence in sentences])
    sentences = pad_sequence(sentences, batch_first=True, padding_value=word_pads[0])
    tags = None if tags[0] is None else pad_sequence(tags, batch_first=True, padding_value=tag_pads[0])
    if sub_words[0] is not None:
        mul_targets = pad_sequence(sub_words, batch_first=True, padding_value=sub_word_pads[0])
        mul_indexes = pad_sequence(token_indice, batch_first=True, padding_value=0)
        other_data = (other_data, mul_targets, mul_indexes)
    return tags, sentences, lengths, len(lengths), actions, gold_spans, gold_btrees, other_data

def build_random_treebank(cfg, echo, data_name, train=False, key=None):
    data_file = f"{cfg.data_root}/" + data_name
    if key is not None:
        data_file = data_file.format(key)
    dataset = ShuffledTreebank(
        data_file,
        batch_size=cfg.batch_size,
        min_length=cfg.min_length, 
        max_length=cfg.max_length, 
    )
    if train and os.path.isfile(f"{cfg.pair_root}/{cfg.pair_name}"):
        old_size = dataset.vocab_size
        vocab_name = cfg.extra_vocab
        if "english" in data_file:
            vocab_name += ".en"
        elif "chinese" in data_file:
            vocab_name += ".zh"
        flag = dataset.expand_vocab(cfg.pair_root, vocab_name)
        if flag: # show expanded vocab
            new_size = dataset.vocab_size
            echo(f"Vocab has been expanded - old: {old_size} vs new: {new_size}")
    if "english" in data_file:
        dataset.tag2idx = STR2DICT(english_tag2idx)
        dataset.idx2tag = SWAP_k_V(dataset.tag2idx) 

    if train:
        sampler = True#SortedBlockSampler
        sampler_def = SortedRandomSampler
        if not isinstance(sampler, bool) and issubclass(sampler, data.Sampler):
            sampler_def = sampler
    else:
        sampler_def = data.sampler.SequentialSampler
    sampler = sampler_def(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False,
        sampler=sampler,
        pin_memory=True, # will change Tuple into List, do NOT rely on these types to determine returned data
        collate_fn=collate_fun
    )
    return data_loader
