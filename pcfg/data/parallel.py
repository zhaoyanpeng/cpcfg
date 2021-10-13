import os, sys
import re, random
import pickle, json
import numpy as np

import torch
import torch.utils.data as data
from transformers import BertTokenizer

from .helper import SortedBlockSampler, SortedRandomSampler, tokenize

class ParallelDataset(data.Dataset):
    """ this is for parallel zh-en data loading, but it can be used for any pair of languages.
    """
    def __init__(
        self, data_file, vocab, tokenizer, vocab_zh, tokenizer_zh, npz_file=None,
        batch_size=128, min_length=1, max_length=100, main_col=1, nsample=float("inf")
    ):
        self.batch_size = batch_size
        self.tokenizer_zh = tokenizer_zh
        self.tokenizer = tokenizer
        self.vocab_zh = vocab_zh
        self.vocab = vocab
        self.main_col = main_col 
        self.mul_captions = list()
        self.mul_indexes = list()
        self.captions = list()
        self.embs = list()
        data_keys = set()
        zh = en = None
        if npz_file is not None:
            npz_data = np.load(npz_file)
            zh, en = npz_data["zh"], npz_data["en"]
            assert zh.shape == en.shape, f"doesn't look like paired embed: |zh| != |en| (|{zh.shape}| != |{en.shape}|)"
        with open(data_file, 'r') as f:
            for iline, line in enumerate(f):
                (sent_zh, sent_en, _, _) = line.split("\t") 
                caption, mul_caption, mul_index = tokenize(
                    sent_en, self.tokenizer, min_length, max_length, lower_case=True
                )
                if len(caption) == 0:
                    continue

                caption_zh, mul_caption_zh, mul_index_zh = tokenize(
                    sent_zh, self.tokenizer_zh, min_length, max_length, lower_case=True
                )
                if len(caption_zh) == 0:
                    continue

                mul_caption = (mul_caption_zh, mul_caption)
                mul_index = (mul_index_zh, mul_index)
                caption = (caption_zh, caption) 

                self.mul_captions.append(mul_caption)
                self.mul_indexes.append(mul_index)
                self.captions.append(caption)
                self.embs.append(
                    (zh[iline], en[iline]) if zh is not None and en is not None else (np.array([]),) * 2
                )
                if len(self.embs) >= nsample:
                    break # 
            if zh is not None:
                assert zh.shape[0] == iline + 1 or len(self.embs) == nsample, \
                    f"doesn't look like aligned embed: |zh| != |samples| (|{zh.shape[0]}| != |{iline + 1}|) "
        self.length = len(self.captions)
        self.indice = list(range(self.length))

    def _shuffle(self):
        indice = torch.randperm(self.length).tolist() 
        indice = sorted(indice, key=lambda k: len(self.captions[k][self.main_col]))
        embs, captions, mul_indexes, mul_captions = list(), list(), list(), list()
        for k in indice:
            embs.append(self.embs[k])
            captions.append(self.captions[k])
            mul_indexes.append(self.mul_indexes[k])
            mul_captions.append(self.mul_captions[k])
        self.embs, self.captions, self.mul_indexes, self.mul_captions = \
            embs, captions, mul_indexes, mul_captions
        self.indice = [self.indice[k] for k in indice]

    def _recover(self):
        indice = np.argsort(self.indice)
        embs, captions, mul_indexes, mul_captions = list(), list(), list(), list()
        for k in indice:
            embs.append(self.embs[k])
            captions.append(self.captions[k])
            mul_indexes.append(self.mul_indexes[k])
            mul_captions.append(self.mul_captions[k])
        self.embs, self.captions, self.mul_indexes, self.mul_captions = \
            embs, captions, mul_indexes, mul_captions
        self.indice = [self.indice[k] for k in indice]

    def __getitem__(self, index):
        def data(col, vocab, tokenizer):
            caption = [vocab(token) for token in self.captions[index][col]]
            caption = torch.tensor(caption)
            mul_caption = tokenizer.convert_tokens_to_ids(self.mul_captions[index][col])
            mul_caption = torch.tensor(mul_caption)
            mul_index = torch.tensor(self.mul_indexes[index][col])
            emb = torch.from_numpy(self.embs[index][col])
            return mul_caption, mul_index, caption, emb
        mul_caption, mul_index, caption, emb = data(1, self.vocab, self.tokenizer)
        mul_caption_zh, mul_index_zh, caption_zh, emb_zh = data(0, self.vocab_zh, self.tokenizer_zh)
        return (mul_caption_zh, mul_caption), (mul_index_zh, mul_index), (caption_zh, caption), (emb_zh, emb), index 

    def __len__(self):
        return self.length

def collate_fun(data):
    mul_captions, mul_indexes, captions, embs, ids = list(zip(*data))
    def data(col):
        lengths = [len(caption[col]) for caption in captions]
        max_len = max(lengths) 
        targets = torch.zeros(len(captions), max_len).long()
        target_indexes = torch.zeros(len(captions), max_len).long()
        for i, cap_len in enumerate(lengths):
            targets[i, : cap_len] = captions[i][col][: cap_len]
            target_indexes[i, : cap_len] = mul_indexes[i][col][: cap_len]

        mul_lengths = [len(mul_caption[col]) for mul_caption in mul_captions]
        mul_max_len = max(mul_lengths)
        mul_targets = torch.zeros(len(captions), mul_max_len).long()
        for i, cap_len in enumerate(mul_lengths):
            mul_targets[i, : cap_len] = mul_captions[i][col][: cap_len]
        lengths = torch.tensor(lengths)
        mul_lengths = torch.tensor(mul_lengths)
        emb_col = torch.stack([emb[col] for emb in embs])
        return targets, lengths, mul_targets, target_indexes, mul_lengths, emb_col 
    targets, lengths, mul_targets, target_indexes, mul_lengths, emb = data(1)
    targets_zh, lengths_zh, mul_targets_zh, target_indexes_zh, mul_lengths_zh, emb_zh = data(0)
    return (
        (targets_zh, lengths_zh, mul_targets_zh, target_indexes_zh, mul_lengths_zh, emb_zh),
        (targets, lengths, mul_targets, target_indexes, mul_lengths, emb), ids
    )

def build_parallel(
    cfg, echo, data_name, vocab, tokenizer, vocab_zh=None, tokenizer_zh=None, train=False, npz_file=None, key=None
):
    vocab_zh = vocab_zh or vocab
    if tokenizer is None or tokenizer_zh is None: # default tokenizer
        def_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    tokenizer = tokenizer or def_tokenizer
    tokenizer_zh = tokenizer_zh or def_tokenizer
    
    main_col = 1 if cfg.lang.lower() == "english" else 0
    pair_file = f"{cfg.pair_root}/{data_name}"
    dataset = ParallelDataset(
        pair_file, vocab, tokenizer, vocab_zh, tokenizer_zh,
        batch_size=cfg.batch_size,
        min_length=cfg.min_length, 
        max_length=cfg.max_length, 
        main_col=main_col,
        npz_file=npz_file,
        nsample=(cfg.train_samples if train else cfg.eval_samples)
    )
    
    if train:
        sampler = SortedBlockSampler
        sampler_def = SortedRandomSampler
        if not isinstance(sampler, bool) and issubclass(sampler, data.Sampler):
            sampler_def = sampler
    else:
        sampler_def = data.sampler.SequentialSampler
    sampler = sampler_def(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=cfg.pair_bsize, 
        shuffle=False,
        sampler=sampler,
        pin_memory=True, 
        collate_fn=collate_fun
    )
    return data_loader
