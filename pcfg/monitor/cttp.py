from omegaconf import OmegaConf
import os, re
from collections import defaultdict

import time
import torch
import numpy as np
from torch import nn

import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import data_parallel
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from ..util import (
    seed_all_rng, numel, get_f1s, update_stats, get_stats, get_tree, get_actions
)
from . import Monitor

def check_grad(model):
    if hasattr(model.text_head, "encoder"):
        text = model.text_head.encoder
    elif hasattr(model.gold_head, "encoder"):
        text = model.gold_head.encoder
    print(text)
    print(text[-1])
    print(text[-1].weight.grad)

    if hasattr(model.pcfg_head, "rule_p_mlp"):
        pcfg = model.pcfg_head.rule_p_mlp #term_mlp #
    elif hasattr(model.pcfg_head, "root_mlp"):
        pcfg = model.pcfg_head.root_mlp #term_mlp #
    print(pcfg)
    print(pcfg[-1])
    print(pcfg[-1].weight.grad)

    import sys; sys.exit()

class Monitor(Monitor):
    """ Contrastive Text-Text Parser.
    """
    def __init__(self, cfg, echo, device):
        super(Monitor, self).__init__(cfg, echo, device)

    def build_cross_iter(self):
        self.iter_count += 1
        self.model.reset() # reset stats
        self.cross_iter = (iter(self.dataloader) if self.cfg.data.rnd_length 
            else iter(np.random.permutation(len(self.dataloader)).tolist())
        )
        self.echo(f"The cross iterator has been reset {self.iter_count}.")

    def recover_data(self):
        for loader in [self.dataloader, self.evalloader, self.pairloader]:
            dataloader = loader.dataset if isinstance(loader, torch.utils.data.DataLoader) else loader
            getattr(dataloader, "_recover", lambda: None)()

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
            self.iter_count = 0
            self.max_length = self.cfg.running.start_max_length
            self.start_time = time.time()
            #self.scaler = torch.cuda.amp.GradScaler()
            #self.save() 
            if self.cfg.data.data_seed is not None: # reset data randomness
                self.echo(f"Random seed ({self.cfg.data.data_seed}) for data sampling.")
                seed_all_rng(self.cfg.data.data_seed)
            self.build_cross_iter()
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
                report = self.infer(self.dataloader, samples=self.cfg.running.eval_samples)
                self.echo(f"{report}")
                return None 
        if self.cfg.running.start_ppl > 0.:
            return self.search_initialization()
        self.echo("Training started...")
        self.last_time = 0.
        self.total_loss = 0
        self.total_step = 0
        self.total_inst = 0
        self.optim_step = 0
        self.iter_count = 0
        self.max_length = self.cfg.running.start_max_length
        self.start_time = time.time()
        #self.scaler = torch.cuda.amp.GradScaler()
        #self.save() 
        if self.cfg.data.data_seed is not None: # reset data randomness
            self.echo(f"Random seed ({self.cfg.data.data_seed}) for data sampling.")
            seed_all_rng(self.cfg.data.data_seed)
        self.build_cross_iter()
        for iepoch in range(self.cfg.optimizer.epochs):
            if isinstance(self.model, DistributedDataParallel):
                self.pairloader.sampler.set_epoch(iepoch)
            if iepoch >= 1:
                pass #break
            self.num_sents = self.num_words = 0.
            self.all_stats = [[0., 0., 0.]]
            self.epoch(iepoch)

    def step(self):
        if self.cfg.running.optim_rate > 1:
            self.model.reduce_grad(cfg.running.optim_rate, sync=False)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.optimizer.max_gnorm
        )      
        self.optimizer.step()
    
    def pre_step(self, step, warmup_step_rate, inc=0):
        force_eval = warmup = False
        return force_eval, warmup 

    def post_step(
        self, iepoch, epoch_step, force_eval, warmup, nchunk, sentences, argmax_btrees=None
    ):
        if (self.cfg.rank == 0 and epoch_step % self.cfg.running.peep_rate == 0):
            self.echo(self.model.stats_main(self.num_sents, self.num_words))
            # example parse
            if argmax_btrees is not None:
                sentence = [self.vocab(word_idx) for word_idx in sentences[0].tolist()]
                pred_tree = get_tree(get_actions(argmax_btrees[0]), sentence)
                self.echo(f"\nPred Tree: {pred_tree}")
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
            self.cfg.running.save_epoch and epoch_step % len(self.pairloader) == 0
        ): # distributed eval
            report = "" ##  parsing perf.
            if self.evalloader is not None:
                self.model.train(False)
                with torch.no_grad():
                    report = self.infer(
                        self.evalloader, samples=self.cfg.data.eval_samples, iepoch=iepoch
                    )
                self.model.train(True)
            if report != "":
                self.echo(f"{report}")
                ppl_criteria = re.search("PPLBound\s(\d+\.\d+)\s", report)
                assert ppl_criteria is not None, f"invalid report: `{report}`"
                ppl_criteria = float(ppl_criteria.group(1))

            report = "" ##  retrieval perf.
            if self.pairloader is not None and self.npz_file is not None:
                self.model.train(False)
                with torch.no_grad():
                    report = self.infer_main(
                        self.pairloader, samples=self.cfg.data.test_samples, iepoch=iepoch
                    )
                self.model.train(True)
            if report != "":
                self.echo(f"{report}")

            if self.cfg.rank == 0:
                self.save()

        # global across epochs 
        if not force_eval and self.optim_step % self.cfg.running.optim_rate == 0: 
            self.model.zero_grad()
            # cross step can only start after the main optim step
            if self.cfg.running.cross_step > 0:
                self.cross_step(iepoch, epoch_step)
        # used for initialization search
        return ppl_criteria 

    def cross_post_step(
        self, iepoch, epoch_step, force_eval, sub_step, sentences, 
        argmax_spans, argmax_btrees, gold_spans, gold_btrees
    ):
        if argmax_btrees is not None:
            for b in range(len(gold_spans)):
                span_b = [(a[0], a[1]) for a in argmax_spans[b] if a[0] != a[1]] #ignore labels
                span_b_set = set(span_b[:-1]) # skip sentence-level constituent
                span_g_set = set([tuple(span) for span in gold_spans[b][:-1]]) # hashable tuple
                update_stats(span_b_set, [span_g_set], self.all_stats)

        if force_eval and (self.cfg.rank == 0 and epoch_step % self.cfg.running.peep_rate == 0):
            msg = self.model.stats(self.num_sents, self.num_words)
            self.echo(f"sub-step {self.optim_step}-{sub_step} {msg} F1: {get_f1s(self.all_stats)[0]:.2f}")
            # example parse
            if argmax_btrees is not None:
                sentence = [self.vocab(word_idx) for word_idx in sentences[0].tolist()]
                pred_tree = get_tree(get_actions(argmax_btrees[0]), sentence)
                gold_tree = get_tree(gold_btrees[0], sentence)
                self.echo(f"\nPred Tree: {pred_tree}\nGold Tree: {gold_tree}")
        # per cross post step
        self.model.zero_grad()
    
    def cross_step(self, iepoch, epoch_step, sub_step=0):
        #for sub_step in range(self.cfg.running.cross_step):
        while sub_step < self.cfg.running.cross_step:
            batch = next(self.cross_iter, None)
            if batch is None:
                self.build_cross_iter()
                batch = next(self.cross_iter, None)
            if isinstance(batch, int):
                batch = self.dataloader[batch]

            tags, sentences, lengths, gold_spans, gold_btrees, sub_words, token_indice = self.make_batch(batch)
            sequences = tags if self.cfg.data.usez_tag else sentences

            max_length = lengths.max()
            if max_length > self.max_length or max_length == 1:
                continue # length < 40 or length filter based on curriculum

            sub_step += 1
            loss, (argmax_spans, argmax_btrees) = self.model(
                sequences, lengths, token_indice=token_indice, sub_words=sub_words
            )
            loss.backward()

            self.step() # optim

            force_eval = sub_step == self.cfg.running.cross_step
            self.cross_post_step(
                iepoch, epoch_step, force_eval, sub_step, sentences, 
                argmax_spans, argmax_btrees, gold_spans, gold_btrees
            )

    def show_batch(self, batch, vocab, tokenizer):
        sentences, lengths, sub_words, token_indice = batch[:4]
        sentences = [
            " ".join([vocab(wid) for i, wid in enumerate(sentence) if i < l])
            for l, sentence in zip(lengths.tolist(), sentences.tolist())
        ]
        self.echo("\n" + "\n".join(sentences))
        if tokenizer is not None:
            sub_lengths = (sub_words != tokenizer.pad_token_id).sum(-1)
            sub_words = [
                " ".join(tokenizer.convert_ids_to_tokens(sentence)[:l])
                for l, sentence in zip(sub_lengths.tolist(), sub_words.tolist())
            ]
            self.echo("\n" + "\n".join(sub_words))

    def epoch(self, iepoch):
        self.model.reset_main()
        ppl_criteria = -1
        all_time = defaultdict(list)
        self.timeit(all_time)        
        device_ids = [i for i in range(self.cfg.num_gpus)]
        nchunk = dist.get_world_size() if torch.distributed.is_initialized() else 1  
        warmup_step_rate = max(self.cfg.optimizer.warmup_steps // 20, 1)
        main_col = 1 if self.cfg.data.lang.lower() == "english" else 0
        def make_batch(batch):
            return tuple(t.to(device=self.device) for t in batch)
        for step, batch in enumerate(self.pairloader, start=iepoch * len(self.pairloader)):
            epoch_step = step % len(self.pairloader) + 1

            batch_x = (batch[1 - main_col][5]).to(device=self.device) # (bsize, dim)
            batch_y = make_batch(batch[main_col]) # sentences, lengths, sub_words, token_indice
            sentences, lengths, sub_words, token_indice = batch_y[:4]

            max_length = lengths.max()
            if max_length > self.max_length or max_length == 1:
                if epoch_step == len(self.pairloader): # special case 
                    ppl_criteria = self.post_step(iepoch, epoch_step, True, False, -1, None)
                continue # length < 40 or length filter based on curriculum

            #self.show_batch(batch_y, self.vocab, self.model.tokenizer())

            self.optim_step += 1 
            bsize = sentences.shape[0]
            force_eval, warmup = self.pre_step(step, warmup_step_rate)

            self.timeit(all_time, key="data")

            loss, (argmax_spans, argmax_btrees) = self.model.forward_main(
                sentences, lengths, token_indice=token_indice, sub_words=sub_words, gold_embs=batch_x
            )

            loss.backward()
            #check_grad(self.model)
            if self.optim_step % self.cfg.running.optim_rate == 0: 
                self.step()

            self.timeit(all_time, key="model")

            self.num_sents += bsize 
            # we implicitly generate </s> so we explicitly count it
            self.num_words += (lengths + 1).sum().item() 

            self.total_step += 1
            self.total_loss += loss.detach()
            self.total_inst += bsize * nchunk

            ppl_criteria = self.post_step(
                iepoch, epoch_step, force_eval, warmup, nchunk, sentences, argmax_btrees=argmax_btrees
            )

            self.timeit(all_time, key="report")

        if not self.cfg.optimizer.use_lars and not self.cfg.optimizer.batch_sch and \
            self.scheduler is not None:
            self.scheduler.step()
        self.max_length = min(self.cfg.running.final_max_length, self.max_length + self.cfg.running.inc_length)
        self.timeit(all_time, show=True)
        return ppl_criteria 

    def infer_main(self, dataloader, samples=float("inf"), iepoch=0):
        if samples <= 0: return ""
        self.model.reset_main()
        num_sents = num_words = 0
        corpus_f1, sentence_f1 = [0., 0., 0.], [] 
        main_col = 1 if self.cfg.data.lang.lower() == "english" else 0

        losses, nsample, nchunk, nbatch = 0, 0, 1, len(dataloader)
        device_ids = [i for i in range(self.cfg.num_gpus)]
        if isinstance(self.model, DistributedDataParallel):
            dataloader.sampler.set_epoch(iepoch)
            nchunk = self.cfg.num_gpus
        peep_rate = max(10, (len(dataloader) // 10))

        def make_batch(batch):
            return tuple(t.to(device=self.device) for t in batch)

        start_time = time.time()
        for ibatch, batch in enumerate(dataloader):
            if nsample >= samples:
                #print(f"{nsample}\t{ibatch}/{nbatch} continue")
                break #continue # iterate through every batch 

            batch_x = (batch[1 - main_col][5]).to(device=self.device) # (bsize, dim)
            batch_y = make_batch(batch[main_col]) # sentences, lengths, sub_words, token_indice
            sentences, lengths, sub_words, token_indice = batch_y[:4]

            max_length = lengths.max()
            if not self.cfg.data.rnd_length and (max_length > 42 or max_length == 1):
                continue # length < 40 or length filter based on curriculum

            bsize = sentences.shape[0]
            sequences = tags if self.cfg.data.usez_tag else sentences

            loss, (argmax_spans, argmax_btrees) = self.model.forward_main(
                sentences, lengths, token_indice=token_indice, sub_words=sub_words, gold_embs=batch_x
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

        msg = self.model.stats_main(num_sents, num_words)
        self.echo(f"{msg}")

        model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        self.echo(f"# sample {nsample}; {nsample / (time.time() - start_time):.2f} samples/s")
        return model.report_main()

