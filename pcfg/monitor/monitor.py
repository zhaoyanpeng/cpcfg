from omegaconf import OmegaConf
import os, re
from collections import defaultdict

import json
import time
import torch
import random
import numpy as np
from torch import nn

import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import data_parallel
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from ..util import (
    seed_all_rng, numel, get_f1s, update_stats, get_stats, get_tree, get_actions, build_parse, VocabContainer
)
from ..data import build_parallel, clean_number
from ..data import build_treebank as build_dataloader
from ..data import build_random_treebank
from ..model import build_main_model
from ..module import LARS, exclude_bias_or_norm, adjust_learning_rate

class Monitor(object):
    def __init__(self, cfg, echo, device):
        super(Monitor, self).__init__()
        self.cfg = cfg
        self.echo = echo
        self.device = device
        self.build_data()
        model = build_main_model(cfg, echo)
        vocab = self.vocab_tag if self.cfg.data.usez_tag else self.vocab
        tunable_params = model.build(**{"vocab": vocab, "num_tag": self.num_tag})
        self.amend_data(model, **tunable_params)
        self.model = DistributedDataParallel(
            model, device_ids=[cfg.rank], find_unused_parameters=True
        ) if torch.distributed.is_initialized() else model 
        self.model.train(not cfg.eval)
        if self.cfg.data.enc_pairs:
            with torch.no_grad():
                self.encode_pairs()
            return # encode sentences 
        self.build_optimizer(tunable_params)

    def reinitialize(self, cfg, echo):
        self.total_init += 1
        random.seed(cfg.seed)
        rnd, irnd = 0, 0 
        while rnd == 0. or irnd < self.total_init + 1e5:
            rnd = random.random() * 1e9
            irnd += 1
        rnd = int(rnd)
        seed_all_rng(rnd) #cfg.seed + rnd) # + rank)
        self.echo(f"Reinitialize everything (model & optimizer) except dataloader: {self.total_init}-th try w/ seed {rnd}.") #cfg.seed + rnd}.")
        #self.build_data()
        #self.recover_data() # TODO may enable
        model = build_main_model(cfg, echo)
        tunable_params = model.build(**{"vocab": self.vocab, "num_tag": self.num_tag})
        #self.amend_data(model)
        self.model = DistributedDataParallel(
            model, device_ids=[cfg.rank], find_unused_parameters=True
        ) if torch.distributed.is_initialized() else model 
        self.model.train(not cfg.eval)
        self.build_optimizer(tunable_params, verbose=False)

    def recover_data(self):
        pass

    def encode_pairs(self):
        def make_batch(batch):
            return tuple(t.to(device=self.device) for t in batch)
        def show_batch(batch, vocab, tokenizer):
            sentences, lengths, sub_words, token_indice, sub_lengths, embs = batch[:6]
            sentences = [
                " ".join([vocab(wid) for i, wid in enumerate(sentence) if i < l]) 
                for l, sentence in zip(lengths.tolist(), sentences.tolist())
            ]
            sub_words = [
                " ".join(tokenizer.convert_ids_to_tokens(sentence)[:l])
                for l, sentence in zip(sub_lengths.tolist(), sub_words.tolist())
            ]
            self.echo("\n" + "\n".join(sentences))
            self.echo("\n" + "\n".join(sub_words))
            self.echo(f"{embs.shape}")
            ## end of show_batch()

        dataset = self.pairloader.dataset
        vocab, tokenizer = dataset.vocab, dataset.tokenizer
        vocab_zh, tokenizer_zh = dataset.vocab_zh, dataset.tokenizer_zh

        zh_emb, en_emb = [], []
        nsample, start_time = 0, time.time()
        peep_rate = max(10, (len(self.pairloader) // 10))
        for ibatch, batch in enumerate(self.pairloader):
            #show_batch(batch[0], vocab_zh, tokenizer_zh)
            #show_batch(batch[1], vocab, tokenizer)

            ## encode sentences
            batch_zh = make_batch(batch[0])[:4]
            h_zh = self.model.encode_text(batch_zh)
            zh_emb.append(h_zh)

            batch_en = make_batch(batch[1])[:4]
            h_en = self.model.encode_text(batch_en)
            en_emb.append(h_en)

            nsample += batch[0][0].shape[0]
            if self.cfg.rank == 0 and (ibatch + 1) % peep_rate == 0:
                self.echo(
                    f"step {ibatch}\t" + 
                    f"{nsample / (time.time() - start_time):.2f} samples/s"
                )
            if ibatch >= 1:
                pass #break
        zh_emb = torch.cat(zh_emb, 0).cpu().numpy()
        en_emb = torch.cat(en_emb, 0).cpu().numpy()
        
        fname = self.cfg.data.pair_name.rsplit(".", 1)[0]
        npz_file = f"{self.cfg.data.pair_root}/{fname}.{self.cfg.model.pcfg.mlm_pooler}"
        self.echo(f"Saving {zh_emb.shape[0]} x 2 sentence vectors to {npz_file}.npz")
        np.savez_compressed(npz_file, zh=zh_emb, en=en_emb)

    def amend_data(self, model, **kwargs):
        tokenizer = model.tokenizer()
        if self.dataloader is not None:
            if not self.cfg.data.rnd_length:
                self.dataloader.tokenizer = tokenizer
                if self.evalloader is not None:
                    self.evalloader.tokenizer = tokenizer
            else:
                self.dataloader.dataset.tokenizer = tokenizer 
                if self.evalloader is not None:
                    self.evalloader.dataset.tokenizer = tokenizer
        if self.cfg.eval:
            self.num_tag = kwargs.get("num_tag", None) 
            self.vocab = kwargs.get("vocab", None) 
            self.echo(f"Amended (evaluation) vocab size: {len(self.vocab)}.")

    def build_data(self):
        if self.cfg.eval:
            self.vocab = self.num_tag = None
            self.dataloader = self.evalloader = self.pairloader = None
            self.evalfile = f"{self.cfg.data.data_root}/{self.cfg.data.eval_name}"
            return
        dcfg = self.cfg.data
        if not dcfg.rnd_length: # change the function
            from ..data import build_treebank as build_dataloader
        else:
            from ..data import build_random_treebank as build_dataloader
        data_name = dcfg.eval_name if self.cfg.eval else dcfg.data_name
        self.dataloader = build_dataloader(dcfg, self.echo, data_name, train=True, key=None)
        dataset = self.dataloader if not dcfg.rnd_length else self.dataloader.dataset
        nstep = len(self.dataloader) 
        nsample = dataset.sents.size(0)
        max_len = dataset.sents.size(1)
        if nstep < self.cfg.running.peep_rate:
            self.cfg.running.peep_rate = nstep 
        self.echo(f"Instantiate main dataloader from `{data_name}': total {nstep} ({nsample}-{self.cfg.running.peep_rate}) batches.")
        self.vocab = VocabContainer(dataset.idx2word, dataset.word2idx)
        self.vocab_tag = VocabContainer(dataset.idx2tag, dataset.tag2idx)
        self.gold_file = f"{dcfg.data_root}/{data_name}"
        # evaluation
        eval_name = "IGNORE_ME" if self.cfg.eval else dcfg.eval_name
        data_path = f"{dcfg.data_root}/{eval_name}"
        do_eval = os.path.isdir(data_path) or os.path.isfile(f"{data_path}") or tf.io.gfile.exists(f"{data_path}")
        self.evalloader = build_dataloader(dcfg, self.echo, eval_name, train=False, key=None) if do_eval else None
        if self.evalloader is not None:
            dataset = self.evalloader if not dcfg.rnd_length else self.evalloader.dataset
            nsample = dataset.sents.size(0)
            max_len = max(dataset.sents.size(1), max_len)
            self.echo(f"Will do evaluation every {self.cfg.running.save_rate} steps on {len(self.evalloader)} ({nsample}) batches.")
            self.gold_file = f"{dcfg.data_root}/{eval_name}"
            dataset.sync_vocab(self.dataloader if not dcfg.rnd_length else self.dataloader.dataset)
        self.echo(f"Vocab size: {dataset.vocab_size}, max sentence length: {max_len}")
        # paired data
        pair_file = f"{dcfg.pair_root}/{dcfg.pair_name}"
        npz_file = f"{dcfg.pair_root}/{dcfg.npz_pair_name}.npz"
        self.npz_file = npz_file if os.path.isfile(npz_file) and not dcfg.enc_pairs else None
        try: # pairloader is not always necessary
            self.pairloader = build_parallel(
                dcfg, self.echo, dcfg.pair_name, self.vocab, tokenizer=None, 
                train=(False if dcfg.enc_pairs else True),
                npz_file=self.npz_file
            ) if os.path.isfile(pair_file) and not self.cfg.eval else None
        except Exception as e:
            self.pairloader = None
        if self.pairloader is not None:
            self.echo(f'Load additional parallel {len(self.pairloader.dataset)} sentences / {len(self.pairloader)} batches.')
        self.num_tag = len(self.dataloader.tag2idx) if dcfg.gold_tag else self.cfg.model.pcfg.num_tag # TODO should be saved in `save()` 

    def search_initialization(self):
        self.echo("Training w/ initialization search started...")
        self.total_init = 0
        ppl = 1e9
        while ppl > self.cfg.running.start_ppl:
            self.echo(f"Current ppl {ppl:.2f} vs required ppl {self.cfg.running.start_ppl:.2f}.")
            escape = True 
            self.last_time = 0.
            self.total_loss = 0
            self.total_step = 0
            self.total_inst = 0
            self.optim_step = 0
            self.max_length = self.cfg.running.start_max_length
            self.start_time = time.time()
            #self.scaler = torch.cuda.amp.GradScaler()
            #self.save() 
            if self.cfg.data.data_seed is not None: # reset data randomness
                self.echo(f"Random seed ({self.cfg.data.data_seed}) for data sampling.")
                seed_all_rng(self.cfg.data.data_seed)
            for iepoch in range(self.cfg.optimizer.epochs):
                if isinstance(self.model, DistributedDataParallel):
                    self.pairloader.sampler.set_epoch(iepoch)
                if iepoch >= 1:
                    pass #break
                self.num_sents = self.num_words = 0.
                self.all_stats = [[0., 0., 0.]]
                ppl = self.epoch(iepoch)
                if iepoch == 0 and ppl > self.cfg.running.start_ppl:
                    escape = False 
                    break
            if not escape:
                self.reinitialize(self.cfg, self.echo)
            else:
                break

    def learn(self):
        if self.cfg.data.enc_pairs:
            return # `encode_pairs` is the only task
        if not self.model.training:
            self.echo("Evaluating started...")
            with torch.no_grad():
                report = self.evaluate(self.evalfile, samples=self.cfg.data.eval_samples)
                self.echo(f"{report}")
                return None 
        if self.cfg.running.start_ppl > 0.:
            return self.search_initialization()
        self.echo("Training started...")
        self.last_time = 0.
        self.total_loss = 0
        self.total_step = 0
        self.total_inst = 0
        self.epoch_inst = 0
        self.optim_step = 0
        self.max_length = self.cfg.running.start_max_length
        self.start_time = time.time()
        #self.scaler = torch.cuda.amp.GradScaler()
        #self.save() 
        if self.cfg.data.data_seed is not None: # reset data randomness
            self.echo(f"Random seed ({self.cfg.data.data_seed}) for data sampling.")
            seed_all_rng(self.cfg.data.data_seed)
        for iepoch in range(self.cfg.optimizer.epochs):
            if isinstance(self.model, DistributedDataParallel):
                self.dataloader.sampler.set_epoch(iepoch)
            if iepoch >= 1:
                pass #break
            self.num_sents = self.num_words = self.epoch_inst = 0.
            self.all_stats = [[0., 0., 0.]]
            self.epoch(iepoch)

    def make_batch(self, batch):
        def show_batch(batch, vocab, vocab_tag, tokenizer):
            tags, sentences, length, batch_size, _, gold_spans, gold_btrees, other_data = batch
            sub_words, token_indice = other_data[1:] if isinstance(other_data[-1], torch.Tensor) else (None, None)
            other_data = other_data[0] # this returns `other_data` for the first sample, we do not need `other_data` though
            lengths = torch.tensor([length] * batch_size, device=self.device).long() if isinstance(length, int) else length

            sentences = [
                " ".join([vocab(wid) for i, wid in enumerate(sentence) if i < l])
                for l, sentence in zip(lengths.tolist(), sentences.tolist())
            ]
            self.echo("\n" + "\n".join(sentences))
            if tags is not None:
                sentences = [
                    " ".join([vocab_tag(wid) for i, wid in enumerate(sentence) if i < l])
                    for l, sentence in zip(lengths.tolist(), tags.tolist())
                ]
                self.echo("\n" + "\n".join(sentences))
            if tokenizer is not None:
                sub_lengths = (sub_words != tokenizer.pad_token_id).sum(-1)
                sub_words = [
                    " ".join(tokenizer.convert_ids_to_tokens(sentence)[:l])
                    for l, sentence in zip(sub_lengths.tolist(), sub_words.tolist())
                ]
                self.echo("\n" + "\n".join(sub_words))
            ## end of show_batch()

        #vocab, vocab_tag, tokenizer = self.vocab, self.vocab_tag, self.model.tokenizer()
        #show_batch(batch, vocab, vocab_tag, tokenizer)
        #import sys; sys.exit(0)

        tags, sentences, length, batch_size, _, gold_spans, gold_btrees, other_data = batch
        sub_words, token_indice = other_data[1:] if isinstance(other_data[-1], torch.Tensor) else (None, None)
        other_data = other_data[0] # this returns `other_data` for the first sample, we do not need `other_data` though

        tags = None if tags is None else tags.to(device=self.device)
        sentences = sentences.to(device=self.device)

        sub_words = sub_words.to(device=self.device) if sub_words is not None else None
        token_indice = token_indice.to(device=self.device) if token_indice is not None else None

        lengths = torch.tensor([length] * batch_size, device=self.device).long() if isinstance(length, int) else length 

        return tags, sentences, lengths, gold_spans, gold_btrees, sub_words, token_indice

    def timeit(self, time_dict, key=None, show=False):
        if self.cfg.rank != 0:
            return 
        if show: # print
            report = ""
            for k, v in time_dict.items():
                report += f"{k} {np.mean(v):.2f} "
            self.echo(f"Time (s): {report.strip()}; # step {self.total_step} # sample {self.total_inst}")
            return
        if key is None: # initialize
            self.last_time = time.time()
        else: # update
            this_time = time.time()
            time_dict[key].append(this_time - self.last_time)
            self.last_time = this_time

    def pre_step(self, step, warmup_step_rate, inc=0):
        force_eval = warmup = False
        return force_eval, warmup 

    def post_step(
        self, iepoch, epoch_step, force_eval, warmup, nchunk, sentences, 
        argmax_spans=None, argmax_btrees=None, gold_spans=None, gold_btrees=None,
    ):
        if argmax_btrees is not None:
            for b in range(len(gold_spans)):
                span_b = [(a[0], a[1]) for a in argmax_spans[b] if a[0] != a[1]] #ignore labels 
                span_b_set = set(span_b[:-1]) # skip sentence-level constituent
                span_g_set = set([tuple(span) for span in gold_spans[b][:-1]]) # hashable tuple
                update_stats(span_b_set, [span_g_set], self.all_stats)

        if force_eval or (self.cfg.rank == 0 and epoch_step % self.cfg.running.peep_rate == 0):
            msg = self.model.stats(self.num_sents, self.num_words)
            self.echo(f"{msg} F1: {get_f1s(self.all_stats)[0]:.2f}")
            # example parse
            if argmax_btrees is not None:
                sentence = [self.vocab(word_idx) for word_idx in sentences[0].tolist()]
                pred_tree = get_tree(get_actions(argmax_btrees[0]), sentence)
                gold_tree = get_tree(gold_btrees[0], sentence)
                # what else do you want to print?
                batch_stats = self.model.batch_stats()
                batch_stats = "" if batch_stats is None else f"\n{batch_stats}"
                self.echo(f"\nPred Tree: {pred_tree}\nGold Tree: {gold_tree}{batch_stats}")
            # overall stats 
            lr_w = self.optimizer.param_groups[0]['lr']
            lr_b = self.optimizer.param_groups[1]['lr']
            self.echo(
                f"epoch {iepoch:>4} step {epoch_step} / {self.total_step}\t" + 
                f"lr_w {lr_w:.2e} lr_b {lr_b:.2e} loss {self.total_loss / self.total_step:.3f} " + 
                f"{self.total_inst / (time.time() - self.start_time):.2f} samples/s" 
            )

        ppl_criteria = -1
        if force_eval or self.total_step % self.cfg.running.save_rate == 0 or (
                self.cfg.running.save_epoch and epoch_step % len(self.dataloader) == 0
            ): # distributed eval
            report = ""
            if self.evalloader is not None:
                self.model.train(False)
                with torch.no_grad():
                    report = self.infer(
                        self.evalloader, samples=self.cfg.data.eval_samples, iepoch=iepoch
                    )
                self.model.train(True)
            if report != "":
                self.echo(f"{report}")
                #ppl_criteria = re.search("PPLBound\s(\d+\.\d+)\s", report)
                ppl_criteria = re.search("PPL:\s(\d+\.\d+)\s", report)
                assert ppl_criteria is not None, f"invalid report: `{report}`"
                ppl_criteria = float(ppl_criteria.group(1))
            if self.cfg.rank == 0:
                self.save()

        # global across epochs 
        if self.optim_step % self.cfg.running.optim_rate == 0: 
            self.model.zero_grad()
        # used for initialization search
        return ppl_criteria 
    
    def step(self):
        if self.cfg.running.optim_rate > 1:
            self.model.reduce_grad(cfg.running.optim_rate, sync=False)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.optimizer.max_gnorm
        )      
        self.optimizer.step()

    def epoch(self, iepoch):
        self.model.reset()
        ppl_criteria = -1
        all_time = defaultdict(list)
        self.timeit(all_time)        
        device_ids = [i for i in range(self.cfg.num_gpus)]
        nchunk = dist.get_world_size() if torch.distributed.is_initialized() else 1  
        warmup_step_rate = max(self.cfg.optimizer.warmup_steps // 20, 1)
        #for step, batch in enumerate(self.dataloader, start=iepoch * len(self.dataloader)):
        def do_batch(batch, step):
            ppl_criteria = -1 # FIXME local variable will not override the global one
            epoch_step = step #% len(self.dataloader)
            tags, sentences, lengths, gold_spans, gold_btrees, sub_words, token_indice = self.make_batch(batch)
            
            max_length = lengths.max()
            if not self.cfg.data.rnd_length and (max_length > self.max_length or max_length == 1):
                if epoch_step == len(self.dataloader): # special case 
                    ppl_criteria = self.post_step(iepoch, epoch_step, True, False, -1, None)
                return #continue # length < 40 or length filter based on curriculum

            self.optim_step += 1 
            bsize = sentences.shape[0]
            sequences = tags if self.cfg.data.usez_tag else sentences
            force_eval, warmup = self.pre_step(step, warmup_step_rate)

            self.timeit(all_time, key="data")

            loss, (argmax_spans, argmax_btrees) = self.model(
                sequences, lengths, token_indice=token_indice, sub_words=sub_words,
                tags=(tags if self.cfg.data.gold_tag else None)
            )
            loss.backward()
            if self.optim_step % self.cfg.running.optim_rate == 0: 
                self.step()

            self.timeit(all_time, key="model")

            self.num_sents += bsize 
            # we implicitly generate </s> so we explicitly count it
            self.num_words += (lengths + 1).sum().item() 

            self.total_step += 1
            self.total_loss += loss.detach()
            self.total_inst += bsize * nchunk
            self.epoch_inst += bsize * nchunk

            ppl_criteria = self.post_step(
                iepoch, epoch_step, force_eval, warmup, nchunk, 
                sentences, argmax_spans, argmax_btrees, gold_spans, gold_btrees
            )

            self.timeit(all_time, key="report")
            return ppl_criteria

        if self.cfg.data.rnd_length:
            for epoch_step, batch_data in enumerate(self.dataloader):
                if epoch_step < 9544:
                    pass #continue
                ppl_criteria = do_batch(batch_data, epoch_step + 1)
                if self.epoch_inst > self.cfg.data.train_samples:
                    ppl_criteria = self.post_step(iepoch, epoch_step, True, False, -1, None)
                    break
        else:
            # iterate over batches
            for epoch_step, ibatch in enumerate(np.random.permutation(len(self.dataloader))):
                if epoch_step < 9980:
                    pass #continue
                batch_data = self.dataloader[ibatch]
                ppl_criteria = do_batch(batch_data, epoch_step + 1)
                if epoch_step > 100:
                    pass #break
                if self.epoch_inst > self.cfg.data.train_samples:
                    ppl_criteria = self.post_step(iepoch, epoch_step, True, False, -1, None)
                    break

        if not self.cfg.optimizer.use_lars and not self.cfg.optimizer.batch_sch and \
            self.scheduler is not None:
            self.scheduler.step()
        self.max_length = min(self.cfg.running.final_max_length, self.max_length + self.cfg.running.inc_length)
        self.timeit(all_time, show=True)
        return ppl_criteria 
        
    def infer(self, dataloader, samples=float("inf"), iepoch=0):
        self.model.reset()
        num_sents = num_words = 0
        corpus_f1, sentence_f1 = [0., 0., 0.], [] 

        losses, nsample, nchunk, nbatch = 0, 0, 1, len(dataloader)
        device_ids = [i for i in range(self.cfg.num_gpus)]
        if isinstance(self.model, DistributedDataParallel):
            dataloader.sampler.set_epoch(iepoch)
            nchunk = self.cfg.num_gpus
        peep_rate = max(10, (len(dataloader) // 10))
        start_time = time.time()
        for ibatch, batch in enumerate(dataloader):
            if nsample >= samples:
                #print(f"{nsample}\t{ibatch}/{nbatch} continue")
                break #continue # iterate through every batch 
            tags, sentences, lengths, gold_spans, gold_btrees, sub_words, token_indice = self.make_batch(batch)
            
            max_length = lengths.max()
            if not self.cfg.data.rnd_length and (max_length > 42 or max_length == 1):
                continue # length < 40 or length filter based on curriculum

            bsize = sentences.shape[0]
            sequences = tags if self.cfg.data.usez_tag else sentences

            loss, (argmax_spans, argmax_btrees) = self.model(
                sequences, lengths, token_indice=token_indice, sub_words=sub_words,
                tags=(tags if self.cfg.data.gold_tag else None)
            )
            nsample += bsize * nchunk
            losses += loss or 0.
            if self.cfg.rank == 0 and (ibatch + 1) % peep_rate == 0:
                self.echo(
                    f"step {ibatch}\t" + #gnorm {grad_norm():.2f} " +
                    f"loss {losses / (ibatch + 1):.5f} " +
                    f"{nsample / (time.time() - start_time):.2f} samples/s"
                )

            num_sents += bsize 
            # we implicitly generate </s> so we explicitly count it
            num_words += (lengths + 1).sum().item() 
            for b in range(bsize):
                span_b = [(a[0], a[1]) for a in argmax_spans[b] if a[0] != a[1]] #ignore labels
                span_b_set = set(span_b[:-1])        
                gold_b_set = set([tuple(span) for span in gold_spans[b][:-1]]) # hashable tuple
                tp, fp, fn = get_stats(span_b_set, gold_b_set) 
                corpus_f1[0] += tp
                corpus_f1[1] += fp
                corpus_f1[2] += fn

                pred_out = span_b_set
                gold_out = gold_b_set
                overlap = pred_out.intersection(gold_out)
                p = float(len(overlap)) / (len(pred_out) + 1e-8)
                r = float(len(overlap)) / (len(gold_out) + 1e-8)
                if len(gold_out) == 0:
                    r = 1. 
                    if len(pred_out) == 0:
                        p = 1.
                f1 = 2 * p * r / (p + r + 1e-8)
                sentence_f1.append(f1)

        corpus_f1 = get_f1s([corpus_f1])[0]
        sentence_f1 = np.mean(np.array(sentence_f1)) * 100
        msg = self.model.stats(num_sents, num_words)
        report = f"\n{msg} Corpus F1: {corpus_f1:.2f} Sentence F1: {sentence_f1:.2f}"
        #self.echo(f"\n{msg} Corpus F1: {corpus_f1:.2f} Sentence F1: {sentence_f1:.2f}")

        model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        self.echo(f"# sample {nsample}; {nsample / (time.time() - start_time):.2f} samples/s")
        #return model.report()
        model.report()
        return report
    
    def evaluate(self, dataloader, samples=float("inf"), iepoch=0):
        self.model.reset()
        num_sents = num_words = 0
        corpus_f1, sentence_f1 = [0., 0., 0.], [] 

        f1_per_label = defaultdict(list) 
        f1_by_length = defaultdict(list)
        
        data_name = (dataloader.rsplit("/", 1)[1]).rsplit(".", 1)[0]
        model_name = self.cfg.model_file.rsplit(".", 1)[0]
        out_prefix = getattr(self.cfg.data, "out_prefix", "default")
        out_file_prefix = f"{self.cfg.model_root}/{self.cfg.model_name}/{model_name}-{out_prefix}-{data_name}"
        pred_out_file = f"{out_file_prefix}.pred"
        gold_out_file = f"{out_file_prefix}.gold"
        pred_out_fw = open(pred_out_file, "w")
        gold_out_fw = open(gold_out_file, "w")

        losses, nsample, nchunk, ibatch = 0, 0, 1, 0
        device_ids = [i for i in range(self.cfg.num_gpus)]
        peep_rate = max(100, (100 // 10))
        start_time = time.time()

        for batch in open(dataloader, "r"):
            if nsample >= samples:
                #print(f"{nsample}\t{ibatch}/{nbatch} continue")
                break #continue # iterate through every batch 

            sentence, gold_spans, labels, tags = json.loads(batch)
            token_indice = sub_words = None
            sentence = sentence.strip().lower().split()
            gold_tree = build_parse(gold_spans, sentence)
            sequence = [clean_number(w) for w in sentence]

            length = max_length = len(sequence)
            if max_length == 1:
                continue # we ignore length 1 sentences
            sequence = self.vocab(sequence)
            sentences = torch.from_numpy(np.array(sequence)).unsqueeze(0)
            sentences = sentences.to(self.device)
            lengths = torch.tensor([length], device=self.device).long() 

            bsize = sentences.shape[0]
            sequences = tags if self.cfg.data.usez_tag else sentences

            loss, (argmax_spans, _) = self.model(
                sequences, lengths, token_indice=token_indice, sub_words=sub_words,
                tags=(tags if self.cfg.data.gold_tag else None)
            )
            nsample += bsize * nchunk
            losses += loss or 0.
            if self.cfg.rank == 0 and (ibatch + 1) % peep_rate == 0:
                self.echo(
                    f"step {ibatch}\t" + #gnorm {grad_norm():.2f} " +
                    f"loss {losses / (ibatch + 1):.5f} " +
                    f"{nsample / (time.time() - start_time):.2f} samples/s"
                )

            ibatch += 1
            num_sents += bsize 
            # we implicitly generate </s> so we explicitly count it
            num_words += (lengths + 1).sum().item() 

            # corpus-level f1
            gold_spans = [(a[0], a[1]) for a in gold_spans]
            pred_spans = [(a[0], a[1]) for a in argmax_spans[0] if a[0] != a[1]]
            pred_span_set = set(pred_spans[:-1]) # remove the trivial sentence-level span
            gold_span_set = set(gold_spans[:-1]) 
            tp, fp, fn = get_stats(pred_span_set, gold_span_set) 
            corpus_f1[0] += tp
            corpus_f1[1] += fp
            corpus_f1[2] += fn
            # sentence-level f1
            pred_out = pred_span_set
            gold_out = gold_span_set
            overlap = pred_out.intersection(gold_out)
            p = float(len(overlap)) / (len(pred_out) + 1e-8)
            r = float(len(overlap)) / (len(gold_out) + 1e-8)
            if len(gold_out) == 0:
                r = 1. 
                if len(pred_out) == 0:
                    p = 1.
            f1 = 2 * p * r / (p + r + 1e-8)
            sentence_f1.append(f1)
            # f1 per label & by length 
            for ispan, gold_span in enumerate(gold_spans[:-1]):
                label = labels[ispan]
                label = re.split("=|-", label)[0]
                f1_per_label.setdefault(label, [0., 0.]) 
                f1_per_label[label][0] += 1

                lspan = gold_span[1] - gold_span[0] + 1
                f1_by_length.setdefault(lspan, [0., 0.])
                f1_by_length[lspan][0] += 1

                if gold_span in pred_span_set:
                    f1_per_label[label][1] += 1 
                    f1_by_length[lspan][1] += 1
            # represent a tree by a binary matrix 
            argmax_tags = torch.zeros(1, length)
            argmax_tree = torch.zeros(1, length, length)
            for l, r, A in argmax_spans[0]:
                argmax_tree[0][l][r] = 1
                if l == r:
                  argmax_tags[0][l] = A 
            # extract the tree from the binary matrix
            argmax_tags = argmax_tags[0].cpu().numpy()
            argmax_tree = argmax_tree[0].cpu().numpy()
            label_matrix = np.zeros((length, length))
            for span in argmax_spans[0]:
                label_matrix[span[0]][span[1]] = span[2]
            pred_tree = {}
            for i in range(length):
                tag = "T-" + str(int(argmax_tags[i].item()) + 1) 
                pred_tree[i] = "(" + tag + " " + sentence[i] + ")"
            for k in np.arange(1, length):
                for s in np.arange(length):
                    t = s + k
                    if t > length - 1: break
                    if argmax_tree[s][t] == 1:
                        nt = "NT-" + str(int(label_matrix[s][t]) + 1)
                        span = "(" + nt + " " + pred_tree[s] + " " + pred_tree[t] +  ")"
                        pred_tree[s] = span
                        pred_tree[t] = span
            # save predicted and gold trees
            pred_tree = pred_tree[0]
            #pred_out_fw.write(pred_tree.strip() + "\n")
            gold_out_fw.write(gold_tree.strip() + "\n")
            saved_item = [sentence, gold_spans, labels, tags, pred_spans, pred_tree.strip()]
            json.dump(saved_item, pred_out_fw)
            pred_out_fw.write("\n")
            #self.echo(pred_tree)
        pred_out_fw.close()
        gold_out_fw.close()

        corpus_f1 = get_f1s([corpus_f1])[0] * 0.01
        sentence_f1 = np.mean(np.array(sentence_f1)) * 1#00
        msg = self.model.stats(num_sents, num_words)
        report = f"\n{msg} Corpus F1: {corpus_f1 * 100:.2f} Sentence F1: {sentence_f1 * 100:.2f}"
        #self.echo(f"\n{msg} Corpus F1: {corpus_f1:.2f} Sentence F1: {sentence_f1:.2f}")

        self.summary_eval(corpus_f1, sentence_f1, f1_per_label, f1_by_length, echo=print)

        model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        self.echo(f"# sample {nsample}; {nsample / (time.time() - start_time):.2f} samples/s")
        #return model.report()
        model.report()
        return report
    
    def summary_eval(self, corpus_f1, sent_f1, f1_per_label, f1_by_length, echo=print):
        f1_ids=["CF1", "SF1", "NP", "VP", "PP", "SBAR", "ADJP", "ADVP"]
        f1s = {"CF1": corpus_f1, "SF1": sent_f1} 

        echo("\nPER-LABEL-F1 (label, acc)\n")
        for k, v in f1_per_label.items():
            echo("{}\t{:.4f} = {}/{}".format(k, v[1] / v[0], v[1], v[0]))
            f1s[k] = v[1] / v[0]

        exist = len([x for x in f1_ids if x in f1s]) == len(f1_ids) 
        if not exist:
            xx = sorted(list(f1_per_label.items()), key=lambda x: -x[1][0])
            f1_ids = ["CF1", "SF1"] + [x[0] for x in xx[:8]]
        f1s = ['{:.2f}'.format(float(f1s[x]) * 100) for x in f1_ids] 

        echo(" ".join(f1_ids))
        echo(" ".join(f1s))

        acc = []

        echo("\nPER-LENGTH-F1 (length, acc)\n")
        xx = sorted(list(f1_by_length.items()), key=lambda x: x[0])
        for k, v in xx:
            echo("{}\t{:.4f} = {}/{}".format(k, v[1] / v[0], v[1], v[0]))
            if v[0] >= 5:
                acc.append((str(k), '{:.2f}'.format(v[1] / v[0])))
        k = [x for x, _ in acc]
        v = [x for _, x in acc]
        #print("\t".join(k))
        #print("\t".join(v))
        echo(" ".join(k))
        echo(" ".join(v))

    def save(self):
        fsave = f"{self.cfg.alias_root}/{self.cfg.alias_name}/{self.total_step:08d}.pth"
        self.echo(f"Saving the checkpoint to {fsave}")
        model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        checkpoint = {
            "cfg": self.cfg, "model": model.collect_state_dict(), "vocab": self.vocab
        }
        torch.save(checkpoint, fsave)

    def build_optimizer(self, tunable_params={}, verbose=True):
        if not self.model.training:
            return
        self.params = (
            list(tunable_params.values())
        )
        for k, v in tunable_params.items():
            if self.cfg.rank == 0:
                pass #self.echo(f"{k} {v.size()}")
        ddp = isinstance(self.model, DistributedDataParallel)
        for k, v in self.model.named_parameters():
            k = re.sub("^module\.", "", k) if ddp else k
            if f"{k}" not in tunable_params:
                v.requires_grad = False
        self.echo(f"# param {numel(self.model) / 1e6:.2f}M # tunable {numel(self.model, True) / 1e6:.2f}M.")
        param_groups = [
            {"params": [p for p in self.params if p.ndim > 1]},
            {"params": [p for p in self.params if p.ndim < 2]},
        ]
        if self.cfg.optimizer.use_lars:
            self.optimizer = LARS(
                param_groups,
                lr=0.,
                weight_decay=self.cfg.optimizer.weight_decay,
                weight_decay_filter=exclude_bias_or_norm,
                lars_adaptation_filter=exclude_bias_or_norm,
            )
        else:
            ocfg = self.cfg.optimizer.optimizer
            scfg = self.cfg.optimizer.scheduler
            self.optimizer = getattr(torch.optim, ocfg[0])(param_groups, **ocfg[1])
            self.scheduler = None if len(scfg) < 2 else getattr(
                torch.optim.lr_scheduler, scfg[0]
            )(self.optimizer, **scfg[1])
        if not self.cfg.verbose or not verbose:
            return
        self.echo(f"Gradienting The Following Parameters:")
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                self.echo(f"{k} {v.size()}")
        self.echo(f"\n{self.model}")

