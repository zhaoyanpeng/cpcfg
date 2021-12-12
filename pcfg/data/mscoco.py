import os, sys, json
import re, random
import pickle, json
import numpy as np
import itertools

import torch
import torch.utils.data as data
from transformers import BertTokenizer

from .helper import SortedBlockSampler, SortedRandomSampler, tokenize

class MscocoDataset(torch.utils.data.Dataset):
    """ this is for visually grounded pcfgs.
    """
    def __init__(
        self, data_file, vocab, tokenizer, train=True, npz_file=None, batch_size=128, 
        embed_dim=0, min_length=1, max_length=100, num_caption_per_image=5, nsample=float("inf")
    ):
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.mul_captions = list()
        self.mul_indexes = list()
        self.captions = list()
        self.labels = list()
        self.spans = list()
        self.tags = list()
        indexes, removed, idx = list(), list(), -1
        with open(data_file, 'r') as f:
            for iline, line in enumerate(f):
                sent, span, label, tag = json.loads(line)
                caption, mul_caption, mul_index = tokenize(
                    sent, self.tokenizer, min_length, max_length, lower_case=True
                )
                if iline // num_caption_per_image == nsample:
                    break # a subset of samples
                if len(caption) == 0: # invalid the sample
                    removed.append((iline, sent))
                    mul_caption = mul_index = caption = label = span = tag = iline = -1
                #sample = (mul_caption, mul_index, caption, label, span, tag)
                self.mul_captions.append(mul_caption)
                self.mul_indexes.append(mul_index)
                self.captions.append(caption)
                self.labels.append(label)
                self.spans.append(span)
                self.tags.append(tag)
                indexes.append(iline)
        self.length = len(self.captions)

        if npz_file is not None:
            self.images = np.load(npz_file)
            self.images = self.images[:int(nsample * num_caption_per_image)]
        else:
            self.images = np.zeros((self.length // num_caption_per_image, embed_dim), dtype=np.float32) 

        if len(removed) > 0: # remove image and all the five captions
            num_caption = self.images.shape[0] * num_caption_per_image
            assert len(indexes) == num_caption, "expected {num_caption} captions for {self.images.shape[0]} images." 
            groups = np.array_split(indexes, self.images.shape[0])
            indice, image_indice = list(), list()
            for igroup, group in enumerate(groups):
                if -1 in group:
                    continue
                indice.extend(group)
                image_indice.append(igroup)
            # update
            self._reorder(indice)
            self.images = self.images[image_indice]
            self.length = len(self.captions)
            assert self.length == self.images.shape[0] * num_caption_per_image
        self.image_indice = np.repeat(range(self.length // num_caption_per_image), num_caption_per_image)

    def __len__(self):
        return self.length

    def _reorder(self, indice):
        mul_captions, mul_indexes, captions, labels, spans, tags = [], [], [], [], [], []
        for k in indice:
            mul_captions.append(self.mul_captions[k])
            mul_indexes.append(self.mul_indexes[k])
            captions.append(self.captions[k])
            labels.append(self.labels[k])
            spans.append(self.spans[k])
            tags.append(self.tags[k])
        self.mul_captions, self.mul_indexes, self.captions, self.labels, self.spans, self.tags = \
            mul_captions, mul_indexes, captions, labels, spans, tags

    def _shuffle(self):
        indice = torch.randperm(self.length).tolist()
        indice = sorted(indice, key=lambda k: len(self.captions[k]))
        self.image_indice = self.image_indice[indice]
        self._reorder(indice)
    
    def __getitem__(self, index):
        image = self.images[self.image_indice[index]]
        caption = [self.vocab[word] for word in self.captions[index]]
        mul_caption = self.tokenizer.convert_tokens_to_ids(self.mul_captions[index])
        mul_index = self.mul_indexes[index]
        return { 
            "image": image,
            "mul_caption": mul_caption,
            "mul_index": mul_index, 
            "caption": caption,
            "label": self.labels[index],
            "span": self.spans[index],
            "tag": self.tags[index],
        } 

class MscocoCollator:
    def __init__(self, device=torch.device("cpu"), tokenizer=None):
        # RuntimeError: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned
        # when pin_memory is true, the collator has to return CPU tensors
        self.device = device
        # TODO tokenizer

    def __call__(self, records):
        union = { 
            k: [record.get(k) for record in records] for k in set().union(*records) 
        } 

        images = np.stack(union["image"], axis=0)
        captions = np.array(
            list(itertools.zip_longest(*union["caption"], fillvalue=0))
        ).T
        mul_captions = np.array(
            list(itertools.zip_longest(*union["mul_caption"], fillvalue=0))
        ).T # filled with self.tokenizer.pad_token_id
        mul_indexes = np.array(
            list(itertools.zip_longest(*union["mul_index"], fillvalue=0))
        ).T
        #spans = np.array(
        #    list(itertools.zip_longest(*union["span"], fillvalue=[0, 0]))
        #).transpose(1, 0, 2)

        return (
            images, captions, mul_captions, mul_indexes, #spans,
            union["span"],
            union["label"], 
            union["tag"]
        )

def build_mscoco(
    cfg, echo, data_name, vocab, tokenizer=None, train=False, 
    num_caption_per_image=5, npz_file=None, embed_dim=0, key=None, nsample=float("inf")
):
    if tokenizer is None: # default tokenizer
        def_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    tokenizer = tokenizer or def_tokenizer
    
    data_file = f"{cfg.data_root}/{data_name}"
    dataset = MscocoDataset(
        data_file, vocab, tokenizer,
        batch_size=cfg.batch_size,
        min_length=cfg.min_length, 
        max_length=(cfg.max_length + (0 if train else 10)),
        num_caption_per_image=num_caption_per_image,
        npz_file=npz_file,
        embed_dim=cfg.embed_dim,
        nsample=nsample,
    )
    
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
        pin_memory=True, 
        collate_fn=MscocoCollator(),
    )
    return data_loader
