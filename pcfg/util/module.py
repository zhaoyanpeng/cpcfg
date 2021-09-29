import torch
from collections import defaultdict

class Stats(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self._stats = defaultdict(float)

    @property
    def stats(self):
        return self._stats

    def __call__(self, **kwargs):
        for k, v in kwargs.items():
            self._stats[k] += v

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = self.count = 0

    def __call__(self, val, n=1):
        self.count += n
        self.sum += val * n

    @property
    def average(self):
        return self.sum / self.count

class Vocab(object):
    def __init__(self):
        self.PAD, self.UNK, self.BOS, self.EOS = ["<pad>", "<unk>", "<s>", "</s>"]
        self.PAD_IDX, self.UNK_IDX, self.BOS_IDX, self.EOS_IDX = [0, 1, 2, 3]
        self.word2idx = {
            self.PAD: self.PAD_IDX,
            self.UNK: self.UNK_IDX,
            self.BOS: self.BOS_IDX,
            self.EOS: self.EOS_IDX,
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def idx(self, token):
        return self.word2idx.get(token, self.word2idx[self.UNK])

    def str(self, idx):
        return self.idx2word.get(idx, self.UNK)

    def has(self, token):
        return token in self.word2idx

    def __getitem__(self, idx):
        return self.idx(idx)

    def __call__(self, key):
        if isinstance(key, int):
            return self.str(key)
        elif isinstance(key, str):
            return self.idx(key)
        elif isinstance(key, list):
            return [self(k) for k in key]
        else:
            raise ValueError(f"type({type(key)}) must be `int` or `str`")

    def __len__(self):
        return len(self.idx2word)

class VocabContainer(Vocab):
    def __init__(self, idx2word, word2idx):
        super(VocabContainer, self).__init__()
        assert len(idx2word) == len(word2idx)
        self.idx2word = idx2word
        self.word2idx = word2idx
