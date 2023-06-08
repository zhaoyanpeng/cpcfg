from omegaconf import OmegaConf
import os, re
from collections import defaultdict

import json
import time
import torch
import numpy as np
from torch import nn

import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import data_parallel
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from ..data import Indexer
from ..data import build_mscoco as build_dataloader
from ..util import (
    seed_all_rng, numel, get_f1s, update_stats, get_stats, get_tree, get_actions, build_parse
)
from . import Monitor

class Monitor(Monitor):
    """ Contrastive Image-Text Parser.
    """
    def __init__(self, cfg, echo, device):
        super(Monitor, self).__init__(cfg, echo, device)
        if cfg.data.num_rnd_consumed <= 0:
            num_rnd_consumed = self.model.count_rnd_consumed()
            self.echo(f"# rnd: {num_rnd_consumed}")
            num_rnd_consumed = sum(num_rnd_consumed.values())
        else: # reset rnd seed and get to the desired rnd-number position for dataloader
            self.echo(f"# rnd: {cfg.data.num_rnd_consumed} # param.: {numel(self.model)}")
            seed_all_rng(cfg.seed)

            #pcfg = self.cfg.model.pcfg
            #kwargs = {"NT": pcfg.num_state, "T": self.num_tag, "vocab": self.vocab}
            #from ..pcfg import build_pcfg_head
            #pcfg = build_pcfg_head(pcfg, **kwargs)
            #self.echo(pcfg.num_rnd_consumed)
            #seed_all_rng(cfg.seed)

            k, step = cfg.data.num_rnd_consumed, 100000
            #k = pcfg.num_rnd_consumed
            for i in range(0, k, step): # consume the first k rnd numbers
                if i + step > k: break
                torch.rand(step)
            torch.rand(k % step)
        #for step, batch in enumerate(self.dataloader, start=0 * len(self.dataloader)):
        #    self.make_batch(batch, show_batch=True)
        #import sys; sys.exit(0)

    def build_data(self):
        if self.cfg.eval and os.path.isfile(self.cfg.data.eval_name):
            self.vocab = self.num_tag = None
            if getattr(self.cfg, "transfer_eval", False): # new vocab for transfer-eval
                self.vocab = Indexer(f"{self.cfg.data.main_vocab}")
                self.echo(f"Vocab size (transfer): {len(self.vocab)}.")
            self.dataloader = self.evalloader = self.testloader = None
            self.evalfile = self.cfg.data.eval_name #f"{self.cfg.data.data_root}/{self.cfg.data.eval_name}"
            return # test MSCOCO parser on WSJ TODO the two datasets have different data roots, so I have to...
        dcfg = self.cfg.data
        # vocab
        self.vocab = Indexer(f"{dcfg.data_root}/{dcfg.main_vocab}")
        self.echo(f"Vocab size: {len(self.vocab)}.")
        # loader 
        def build_loader(
            data_name, msg, train=False, eval=False, test=False,
            num_caption_per_image=5, nsample=float("inf")
        ):
            data_path = f"{dcfg.data_root}/{data_name}"
            npz_file = f"{dcfg.data_root}/{data_name.split('_', 1)[0]}_{dcfg.npz_token}.npy"
            npz_file = npz_file if os.path.isfile(npz_file) and dcfg.embed_dim > 0 else None
            self.echo(f"Pre-computed image vectors: {npz_file}")
            dataloader = build_dataloader(
                dcfg, self.echo, data_name, self.vocab, train=train, nsample=nsample,
                npz_file=npz_file, num_caption_per_image=num_caption_per_image,
            ) if os.path.isfile(f"{data_path}") else None
            if dataloader is None:
                return None # break
            nstep = len(dataloader) 
            nsample = len(dataloader.dataset)
            if train and nstep < self.cfg.running.peep_rate:
                self.cfg.running.peep_rate = nstep 
            info = (
                f"Instantiate {msg} dataloader from `{data_name}': total {nstep} ({nsample}-{self.cfg.running.peep_rate}) batches." 
                if train else f"Will do {msg} every {self.cfg.running.save_rate} steps on {len(dataloader)} ({nsample}) batches."
            )
            self.echo(info)
            if train or eval: # used for what?
                self.gold_file = data_path  
            elif test:
                self.gold_file_test = data_path  
            return dataloader 
        num_caption_per_image = 5 if dcfg.vis_mode else 1
        # train (main loader)
        data_name = dcfg.eval_name if self.cfg.eval else dcfg.data_name
        self.dataloader = build_loader(
            data_name, "main", train=True, nsample=dcfg.train_samples, num_caption_per_image=num_caption_per_image
        ) if not self.cfg.eval else build_loader(
            data_name, "main", test= True, nsample= dcfg.eval_samples, num_caption_per_image=num_caption_per_image
        )
        # evaluation
        eval_name = "IGNORE_ME" if self.cfg.eval else dcfg.eval_name
        self.evalloader = build_loader(
            eval_name, "eval", eval=True, nsample=dcfg.eval_samples, num_caption_per_image=num_caption_per_image
        )
        # test 
        test_name = "IGNORE_ME" if self.cfg.eval else dcfg.test_name
        self.testloader = build_loader(
            test_name, "test", test=True, nsample=dcfg.test_samples, num_caption_per_image=num_caption_per_image
        )
        # necessary for model building
        self.num_tag = None if dcfg.gold_tag else self.cfg.model.pcfg.num_tag # TODO should be saved in `save()`

    def build_cross_iter(self):
        pass

    def recover_data(self):
        pass

    def search_initialization(self):
        pass

    def show_batch(self, batch, vocab, tokenizer):
        images, sentences, sub_words, token_indice, spans, labels, tags = batch
        lengths = (sentences != vocab.PAD_IDX).sum(-1)
        sentences = [
            " ".join([vocab(wid) for i, wid in enumerate(sentence) if i < l])
            for l, sentence in zip(lengths.tolist(), sentences.tolist())
        ]
        self.echo("\n" + "\n".join(sentences))
        if tokenizer is not None:
            #print(tokenizer.pad_token_id)
            sub_lengths = (sub_words != tokenizer.pad_token_id).sum(-1)
            sub_words = [
                " ".join(tokenizer.convert_ids_to_tokens(sentence)[:l])
                for l, sentence in zip(sub_lengths.tolist(), sub_words.tolist())
            ]
            self.echo("\n" + "\n".join(sub_words))
        self.echo(spans)

    def make_batch(self, batch, show_batch=False):
        images, sentences, sub_words, token_indice, spans, labels, tags = batch
        items = tuple(torch.tensor(t, device=self.device) for t in batch[:4])
        if show_batch:
            self.show_batch(batch, self.vocab, self.dataloader.dataset.tokenizer) #self.model.tokenizer())
        return items + tuple(batch[4:]) 

    def learn(self):
        if self.cfg.data.enc_pairs:
            return # `encode_pairs` is the only task
        if not self.model.training:
            self.echo("Evaluating started...")
            with torch.no_grad():
                if self.dataloader is not None:
                    self.echo(f"On MSCOCO... {self.gold_file_test}")
                    report = self.evaluate_coco(self.dataloader, samples=self.cfg.data.eval_samples)
                else: # WSJ or something else?
                    self.echo(f"On Others... {self.evalfile}")
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
        self.optim_step = 0
        self.iter_count = 0
        self.max_length = self.cfg.running.start_max_length
        self.start_time = time.time()
        #self.scaler = torch.cuda.amp.GradScaler()
        self.save() 
        if self.cfg.data.data_seed is not None: # reset data randomness
            self.echo(f"Random seed ({self.cfg.data.data_seed}) for data sampling.")
            seed_all_rng(self.cfg.data.data_seed)
        self.build_cross_iter()
        for iepoch in range(self.cfg.optimizer.epochs):
            if isinstance(self.model, DistributedDataParallel):
                self.dataloader.sampler.set_epoch(iepoch)
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
        self, iepoch, epoch_step, force_eval, warmup, nchunk, sentences, 
        argmax_spans=None, argmax_btrees=None, gold_spans=None,
    ):
        if argmax_btrees is not None:
            for b in range(len(gold_spans)):
                span_b = [(a[0], a[1]) for a in argmax_spans[b] if a[0] != a[1]] #ignore labels 
                span_b_set = set(span_b[:-1]) # skip sentence-level constituent
                span_g_set = set([tuple(span) for span in gold_spans[b][:-1]]) # hashable tuple
                update_stats(span_b_set, [span_g_set], self.all_stats)

        if (self.cfg.rank == 0 and epoch_step % self.cfg.running.peep_rate == 0):
            msg = self.model.stats_main(self.num_sents, self.num_words)
            self.echo(f"{msg} F1: {get_f1s(self.all_stats)[0]:.2f}")
            # example parse
            if argmax_btrees is not None:
                sentence = [self.vocab(word_idx) for word_idx in sentences[0].tolist() if word_idx != self.vocab.PAD_IDX]
                pred_tree = get_tree(get_actions(argmax_btrees[0]), sentence)
                gold_tree = build_parse(gold_spans[0], sentence)
                self.echo(f"\nPred Tree: {pred_tree}\nGold Tree: {gold_tree}")
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
            report = "" ##  parsing perf. and retrieval perf.
            if self.evalloader is not None:
                self.model.train(False)
                with torch.no_grad():
                    report = self.infer_main(
                        self.evalloader, samples=self.cfg.data.eval_samples, iepoch=iepoch
                    )
                self.model.train(True)
            if report != "":
                self.echo(f"{report}")
                ppl_criteria = re.search("PPLBound\s(\d+\.\d+)\s", report)
                assert ppl_criteria is not None, f"invalid report: `{report}`"
                ppl_criteria = float(ppl_criteria.group(1))

            report = "" ##  parsing perf. and retrieval perf.
            if self.testloader is not None:
                self.model.train(False)
                with torch.no_grad():
                    report = self.infer_main(
                        self.testloader, samples=self.cfg.data.test_samples, iepoch=iepoch
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
                pass #self.cross_step(iepoch, epoch_step)
        # used for initialization search
        return ppl_criteria 

    def epoch(self, iepoch):
        self.model.reset_main()
        ppl_criteria = -1
        all_time = defaultdict(list)
        self.timeit(all_time)        
        device_ids = [i for i in range(self.cfg.num_gpus)]
        nchunk = dist.get_world_size() if torch.distributed.is_initialized() else 1  
        warmup_step_rate = max(self.cfg.optimizer.warmup_steps // 20, 1)
        for step, batch in enumerate(self.dataloader, start=iepoch * len(self.dataloader)):
            epoch_step = step % len(self.dataloader) + 1
            
            images, sentences, sub_words, token_indice, gold_spans, _, _ = self.make_batch(batch)
            lengths = (sentences != self.vocab.PAD_IDX).sum(-1)

            max_length = lengths.max()
            if max_length > self.max_length or max_length == 1:
                if epoch_step == len(self.dataloader): # special case 
                    ppl_criteria = self.post_step(iepoch, epoch_step, True, False, -1, None)
                continue # length < 40 or length filter based on curriculum

            #self.show_batch(batch_y, self.vocab, self.model.tokenizer())

            self.optim_step += 1 
            bsize = sentences.shape[0]
            force_eval, warmup = self.pre_step(step, warmup_step_rate)

            self.timeit(all_time, key="data")

            loss, (argmax_spans, argmax_btrees) = self.model.forward_main(
                sentences, lengths, token_indice=token_indice, sub_words=sub_words, gold_embs=images
            )

            loss.backward()
            #check_grad(self.model)
            if self.optim_step % self.cfg.running.optim_rate == 0: 
                self.step() # zero grad in `post_step`

            self.timeit(all_time, key="model")

            self.num_sents += bsize 
            # we implicitly generate </s> so we explicitly count it
            self.num_words += (lengths + 1).sum().item() 

            self.total_step += 1
            self.total_loss += loss.detach()
            self.total_inst += bsize * nchunk

            ppl_criteria = self.post_step(
                iepoch, epoch_step, force_eval, warmup, nchunk, sentences, 
                argmax_spans=argmax_spans, argmax_btrees=argmax_btrees, gold_spans=gold_spans
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

        losses, nsample, nchunk, nbatch = 0, 0, 1, len(dataloader)
        device_ids = [i for i in range(self.cfg.num_gpus)]
        if isinstance(self.model, DistributedDataParallel):
            dataloader.sampler.set_epoch(iepoch)
            nchunk = self.cfg.num_gpus
        peep_rate = max(10, (len(dataloader) // 10))
        num_x2_per_x1 = 5

        start_time = time.time()
        for ibatch, batch in enumerate(dataloader):
            if nsample >= samples * num_x2_per_x1:
                #print(f"{nsample}\t{ibatch}/{nbatch} continue")
                break #continue # iterate through every batch 

            images, sentences, sub_words, token_indice, gold_spans, _, _ = self.make_batch(batch)
            lengths = (sentences != self.vocab.PAD_IDX).sum(-1)

            max_length = lengths.max()
            if max_length > 50 or max_length == 1:
                continue # length < 50 or length filter

            bsize = sentences.shape[0]
            sequences = tags if self.cfg.data.usez_tag else sentences

            loss, (argmax_spans, argmax_btrees) = self.model.forward_main(
                sentences, lengths, token_indice=token_indice, sub_words=sub_words, gold_embs=images
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
        msg = self.model.stats_main(num_sents, num_words)
        report = f"\n{msg} Corpus F1: {corpus_f1:.2f} Sentence F1: {sentence_f1:.2f}"

        model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        self.echo(f"# sample {nsample}; {nsample / (time.time() - start_time):.2f} samples/s")

        self.echo(model.report_main(num_x2_per_x1=num_x2_per_x1)) # retrieval perf.
        return report

    def evaluate_coco(self, dataloader, samples=float("inf"), iepoch=0):
        self.model.reset()
        num_sents = num_words = 0
        corpus_f1, sentence_f1 = [0., 0., 0.], []

        f1_per_label = defaultdict(list)
        f1_by_length = defaultdict(list)

        data_name = (self.gold_file_test.rsplit("/", 1)[1]).rsplit(".", 1)[0]
        model_name = self.cfg.model_file.rsplit(".", 1)[0]
        out_file_prefix = f"{self.cfg.model_root}/{self.cfg.model_name}/{model_name}-{data_name}"
        pred_out_file = f"{out_file_prefix}.pred"
        gold_out_file = f"{out_file_prefix}.gold"
        pred_out_fw = open(pred_out_file, "w")
        gold_out_fw = open(gold_out_file, "w")

        losses, nsample, nchunk, ibatch = 0, 0, 1, 0
        device_ids = [i for i in range(self.cfg.num_gpus)]
        peep_rate = max(10, (len(dataloader) // 10))
        start_time = time.time()

        for ibatch, batch in enumerate(dataloader):
            if nsample >= samples:
                #print(f"{nsample}\t{ibatch}/{nbatch} continue")
                break #continue # iterate through every batch

            images, sentences, sub_words, token_indice, \
                batch_gold_spans, batch_labels, batch_tags = self.make_batch(batch)
            lengths = (sentences != self.vocab.PAD_IDX).sum(-1)

            bsize = sentences.shape[0]
            loss, (argmax_spans, _) = self.model(
                sentences, lengths, token_indice=token_indice, sub_words=sub_words, use_mean=True
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
            for b, (length, sentence) in enumerate(zip(lengths.tolist(), sentences.tolist())):
                sentence = [
                    self.vocab(wid) for i, wid in enumerate(sentence) if i < length
                ] # ideally, sentence should not be number-cleaned, but ...

                tags = batch_tags[b]
                labels = batch_labels[b]
                gold_spans = batch_gold_spans[b]

                gold_tree = build_parse(gold_spans, sentence)

                # corpus-level f1
                gold_spans = [(a[0], a[1]) for a in gold_spans if a[0] != a[1]]
                pred_spans = [(a[0], a[1]) for a in argmax_spans[b] if a[0] != a[1]]
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
                for l, r, A in argmax_spans[b]:
                    argmax_tree[0][l][r] = 1
                    if l == r:
                      argmax_tags[0][l] = A
                # extract the tree from the binary matrix
                argmax_tags = argmax_tags[0].cpu().numpy()
                argmax_tree = argmax_tree[0].cpu().numpy()
                label_matrix = np.zeros((length, length))
                for span in argmax_spans[b]:
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
                #self.echo(pred_tree) # end loop over batch
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
