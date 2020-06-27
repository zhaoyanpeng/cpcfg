#!/usr/bin/env python3
import sys
import os

import argparse
import json
import random
import shutil
import copy

import torch
from torch import cuda
import numpy as np
import time
import logging
from data import Dataset
from PCFG import PCFG
from utils import *
from models import CompPCFG
from torch.nn.init import xavier_uniform_

from torch_struct import SentCFG
from torch_struct.networks import CPCFG, RoughCCFG, CompoundCFG
from torch_struct.networks import ACompPCFG as YoonPCFG


mappings = {
    "t_emb": "term_emb",
    "nt_emb": "nonterm_emb",
    "root_mlp.1.lin1": "root_mlp.1.linear.0",
    "root_mlp.1.lin2": "root_mlp.1.linear.2",
    "root_mlp.2.lin1": "root_mlp.2.linear.0",
    "root_mlp.2.lin2": "root_mlp.2.linear.2",
    "enc_params": "enc_out",
    "vocab_mlp.0": "term_mlp.0",
    "vocab_mlp.1.lin1": "term_mlp.1.linear.0",
    "vocab_mlp.1.lin2": "term_mlp.1.linear.2",
    "vocab_mlp.2.lin1": "term_mlp.2.linear.0",
    "vocab_mlp.2.lin2": "term_mlp.2.linear.2",
    "vocab_mlp.3": "term_mlp.3",
}

def create_state_dict(model, mappings):
    dict_items = dict()
    new_mappings = dict()
    for k, v in model.state_dict().items():
        name_fields = k.split(".")
        key = ".".join(name_fields[:-1])
        key = name_fields[-1] if key == "" else key
        if key in mappings:
            key = mappings[key]
            if len(name_fields) > 1:
                key = "{}.{}".format(key, name_fields[-1])
        else:
            key = k
        dict_items[key] = v
        new_mappings[k] = key 
    return dict_items, new_mappings

parser = argparse.ArgumentParser()

# Data path options
parser.add_argument('--train_file', default='data/ptb-train.pkl')
parser.add_argument('--val_file', default='data/ptb-val.pkl')
parser.add_argument('--save_path', default='compound-pcfg.pt', help='where to save the model')
parser.add_argument('--model_type', default='1st', type=str, help='model name (1st/2nd)')
parser.add_argument('--infer_fast', default=True, type=bool, help='model name (1st/2nd)')
parser.add_argument('--model_init', default=None, type=str, help='load a model')
parser.add_argument('--perturb_it', default=False, type=str, help='perturb parameter values')
parser.add_argument('--data_random', default=False, type=bool, help='sync randomness for data fetching')
parser.add_argument('--keep_random', default=False, type=bool, help='same randomness for data fetching')
# Model options
# Generative model parameters
parser.add_argument('--z_dim', default=64, type=int, help='latent dimension')
parser.add_argument('--t_states', default=60, type=int, help='number of preterminal states')
parser.add_argument('--nt_states', default=30, type=int, help='number of nonterminal states')
parser.add_argument('--state_dim', default=256, type=int, help='symbol embedding dimension')
# Inference network parameters
parser.add_argument('--h_dim', default=512, type=int, help='hidden dim for variational LSTM')
parser.add_argument('--w_dim', default=512, type=int, help='embedding dim for variational LSTM')
#
parser.add_argument('--share_term', default=False, type=bool, help='share preterminal rules')
parser.add_argument('--share_rule', default=False, type=bool, help='share binary rules')
parser.add_argument('--share_root', default=False, type=bool, help='share binary rules')
parser.add_argument('--wo_enc_emb', default=False, type=bool, help='how to encode z')
parser.add_argument('--use_mean', default=False, type=bool, help='how to use z')
# Optimization options
parser.add_argument('--num_epochs', default=10, type=int, help='number of training epochs')
parser.add_argument('--lr', default=0.001, type=float, help='starting learning rate')
parser.add_argument('--max_grad_norm', default=3, type=float, help='gradient clipping parameter')
parser.add_argument('--max_length', default=30, type=float, help='max sentence length cutoff start')
parser.add_argument('--len_incr', default=1, type=int, help='increment max length each epoch')
parser.add_argument('--final_max_length', default=40, type=int, help='final max length cutoff')
parser.add_argument('--beta1', default=0.75, type=float, help='beta1 for adam')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2 for adam')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--seed', default=3435, type=int, help='random seed')
parser.add_argument('--print_every', type=int, default=1000, help='print stats after N batches')
# Regularization strength
parser.add_argument('--reg_h_alpha', default=0., type=float, help='entropy of PCFG')
parser.add_argument('--reg_d_alpha', default=0., type=float, help='prob. dist. of rules of diff. LHS')
parser.add_argument('--reg_c_alpha', default=0., type=float, help='prob. dist. of rules of the same LHS')

def main(args, print):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  train_data = Dataset(args.train_file)
  val_data = Dataset(args.val_file)  
  train_sents = train_data.batch_size.sum()
  vocab_size = int(train_data.vocab_size)    
  max_len = max(val_data.sents.size(1), train_data.sents.size(1))
  print('Train: %d sents / %d batches, Val: %d sents / %d batches' % 
        (train_data.sents.size(0), len(train_data), val_data.sents.size(0), len(val_data)))
  print('Vocab size: %d, Max Sent Len: %d' % (vocab_size, max_len))
  print('Save Path {}'.format(args.save_path))
  cuda.set_device(args.gpu)
  
  if args.model_type == '1st':
      model = RoughCCFG  
  elif args.model_type == '2nd':
      model = CompoundCFG 
  elif args.model_type == '3rd':
      model = YoonPCFG 
  elif args.model_type == '4th':
      model = CPCFG
  else:
      raise NameError("Invalid parser type: {}".format(opt.parser_type)) 
  kwarg = {
      "share_term": getattr(args, "share_term", False),
      "share_rule": getattr(args, "share_rule", False),
      "share_root": getattr(args, "share_root", False),
      "wo_enc_emb": getattr(args, "wo_enc_emb", False),
  }
  if True:
    model = model(vocab_size, args.nt_states, args.t_states, 
                  h_dim = args.h_dim,
                  w_dim = args.w_dim,
                  z_dim = args.z_dim,
                  s_dim = args.state_dim, **kwarg)
    if args.model_type == '2nd' and args.model_init:
        global mappings
        checkpoint = torch.load(args.model_init, map_location='cpu')
        init_model = checkpoint["model"]
        state_dict, mappings = create_state_dict(init_model, mappings)
        model.update_state_dict(state_dict)

        old_state_dict = model.state_dict()
        for k, v in init_model.named_parameters():
            key = mappings[k]
            v2 = old_state_dict[key]
            eq = torch.equal(v, v2)
            print("P: {} -> {} {} {}".format(k, key, v.size(), eq))
    elif args.model_init:
        checkpoint = torch.load(args.model_init, map_location='cpu')
        init_model = checkpoint["model"]

        def perturb_params(old_model, new_model):
            old_state_dict = old_model.state_dict()
            new_state_dict = new_model.state_dict()
            for k, v1 in new_model.named_parameters():
                v2 = old_state_dict[k]
                #v0 = v2.data * 0.4 + v1.data * 0.6
                v0 = v2.data * v1.data
                v1.data.copy_(v0) 
                #eq = torch.equal(new_state_dict[k], v1)
                #print("P: {} {}".format(k, eq))

        if not args.perturb_it:
            print("override parser's params.")
            model.load_state_dict(init_model.state_dict())
        else:
            print("perturb parser's params.")
            perturb_params(init_model, model)
  else:
    from torch_struct.networks import CompPCFG as model 
    model = model(vocab = vocab_size,
                  state_dim = args.state_dim,
                  t_states = args.t_states,
                  nt_states = args.nt_states,
                  h_dim = args.h_dim,
                  w_dim = args.w_dim,
                  z_dim = args.z_dim)
    for name, param in model.named_parameters():    
      if param.dim() > 1:
        xavier_uniform_(param)

    old_state_dict = model.state_dict()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from torch_struct.networks import ACompPCFG as model 
    model = model(vocab_size, args.nt_states, args.t_states, 
                  h_dim = args.h_dim,
                  w_dim = args.w_dim,
                  z_dim = args.z_dim,
                  s_dim = args.state_dim)

    for k, v in model.named_parameters():
        v2 = old_state_dict[k]
        eq = torch.equal(v, v2)
        print("P: {} -> {} {} {}".format(k, "", v.size(), eq))

  pcfg = None if args.infer_fast else PCFG(args.nt_states, args.t_states)

  fsave = args.save_path + "/{}.pt".format(0)
  print('Saving {} checkpoint to {}'.format(0, fsave))
  checkpoint = {
    'args': args.__dict__,
    'model': model.cpu(),
    'word2idx': train_data.word2idx,
    'idx2word': train_data.idx2word
  }
  torch.save(checkpoint, fsave)

  print("model architecture")
  print(model)
  model.train()
  model.cuda()
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas = (args.beta1, args.beta2))
  best_val_ppl = 1e5
  best_val_f1 = 0
  epoch = 0

  #val_ppl, val_f1 = eval(val_data, model, print, pcfg)
  #import sys
  #sys.exit(0)
  
  if args.data_random: # reset data randomness
    seed = 1847 if args.keep_random else args.seed
    print("random seed {} for data shuffling.".format(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)

  while epoch < args.num_epochs:
    start_time = time.time()
    epoch += 1  
    print('Starting epoch %d' % epoch)
    train_nll = 0.
    train_kl = 0.
    num_sents = 0.
    num_words = 0.
    all_stats = [[0., 0., 0.]]
    b = 0
    for i in np.random.permutation(len(train_data)):
      b += 1
      sents, length, batch_size, _, gold_spans, gold_binary_trees, _ = train_data[i]      
      """ 
      print(i)
      if b > 10: import sys; sys.exit() 
      sent_str = [train_data.idx2word[word_idx] for word_idx in list(sents[0].cpu().numpy())]
      if b == 1000:
        print("Gold Tree: %s" % get_tree(gold_binary_trees[0], sent_str))
      #continue
      """ 
      if length > args.max_length or length == 1: #length filter based on curriculum 
        continue
      sents = sents.cuda()
      optimizer.zero_grad()
      """
      nll, kl, binary_matrix, argmax_spans = model(sents, argmax=True)      
      (nll+kl).mean().backward()
      """
      lengths = torch.tensor([length] * batch_size, device=sents.device).long() 
      params, kl = model(sents, lengths)
      if args.infer_fast: 
        dist = SentCFG(params, lengths=lengths)
        spans = dist.argmax[-1]
        argmax_spans, tree = extract_parses(spans, lengths.tolist(), inc=0)
        nll = -dist.partition
      else:
        log_Z = pcfg._inside(*params)
        nll = -log_Z
        with torch.no_grad():
          max_score, binary_matrix, argmax_spans = pcfg._viterbi(*params)

      #print(kl.tolist())
      #print(nll.tolist())
      #print(" ".join(sent_str))

      kl = torch.zeros_like(nll) if kl is None else kl
      (nll + kl).mean().backward()
      train_nll += nll.sum().item()
      train_kl += kl.sum().item()
      torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)      
      optimizer.step()
      num_sents += batch_size
      num_words += batch_size * (length + 1) # we implicitly generate </s> so we explicitly count it
      for bb in range(batch_size):
        span_b = [(a[0], a[1]) for a in argmax_spans[bb] if a[0] != a[1]] #ignore labels
        span_b_set = set(span_b[:-1])
        update_stats(span_b_set, [set(gold_spans[bb][:-1])], all_stats)
      if b % args.print_every == 0:
        all_f1 = get_f1(all_stats)
        param_norm = sum([p.norm()**2 for p in model.parameters()]).item()**0.5
        gparam_norm = sum([p.grad.norm()**2 for p in model.parameters() 
                           if p.grad is not None]).item()**0.5
        log_str = 'Epoch: %d, Batch: %d/%d, |Param|: %.6f, |GParam|: %.2f,  LR: %.4f, ' + \
                  'ReconPPL: %.2f, KL: %.4f, PPLBound: %.2f, ValPPL: %.2f, ValF1: %.2f, ' + \
                  'CorpusF1: %.2f, Throughput: %.2f examples/sec'
        print(log_str %
              (epoch, b, len(train_data), param_norm, gparam_norm, args.lr, 
               np.exp(train_nll / num_words), train_kl /num_sents, 
               np.exp((train_nll + train_kl)/num_words), best_val_ppl, best_val_f1, 
               all_f1[0], num_sents / (time.time() - start_time)))
        # print an example parse
        if not args.infer_fast: 
          tree = get_tree_from_binary_matrix(binary_matrix[0], length)
          action = get_actions(tree)
        else:
          action = get_actions(tree[0])
        sent_str = [train_data.idx2word[word_idx] for word_idx in list(sents[0].cpu().numpy())]
        print("Pred Tree: %s" % get_tree(action, sent_str))
        print("Gold Tree: %s" % get_tree(gold_binary_trees[0], sent_str))
      #break
    args.max_length = min(args.final_max_length, args.max_length + args.len_incr)
    print('--------------------------------')
    print('Checking validation perf...')    
    val_ppl, val_f1 = eval(val_data, model, print, pcfg)
    print('--------------------------------')

    fsave = args.save_path + "/{}.pt".format(epoch)
    print('Saving {} checkpoint to {}'.format(epoch, fsave))
    checkpoint = {
      'args': args.__dict__,
      'model': model.cpu(),
      'word2idx': train_data.word2idx,
      'idx2word': train_data.idx2word
    }
    torch.save(checkpoint, fsave)
    model.cuda()

    if val_ppl < best_val_ppl:
      best_val_ppl = val_ppl
      best_val_f1 = val_f1
      checkpoint = {
        'args': args.__dict__,
        'model': model.cpu(),
        'word2idx': train_data.word2idx,
        'idx2word': train_data.idx2word
      }
      print('Saving the best checkpoint to %s' % args.save_path + "/best.pt")
      torch.save(checkpoint, args.save_path + "/best.pt")
      model.cuda()
    #break

def eval(data, model, print, pcfg):
  model.eval()
  num_sents = 0
  num_words = 0
  total_nll = 0.
  total_kl = 0.
  corpus_f1 = [0., 0., 0.] 
  sent_f1 = [] 
  with torch.no_grad():
    for i in range(len(data)):
      sents, length, batch_size, _, gold_spans, gold_binary_trees, other_data = data[i] 
      if length == 1:
        continue
      sents = sents.cuda()
      # note that for unsuperised parsing, we should do model(sents, argmax=True, use_mean = True)
      # but we don't for eval since we want a valid upper bound on PPL for early stopping
      # see eval.py for proper MAP inference
      """
      nll, kl, binary_matrix, argmax_spans = model(sents, argmax=True)
      """
      lengths = torch.tensor([length] * batch_size, device=sents.device).long() 
      params, kl = model(sents, lengths)
      if args.infer_fast: 
        dist = SentCFG(params, lengths=lengths)
        spans = dist.argmax[-1]
        argmax_spans, tree = extract_parses(spans, lengths.tolist(), inc=0)
        nll = -dist.partition
      else:
        log_Z = pcfg._inside(*params)
        nll = -log_Z
        with torch.no_grad():
          max_score, binary_matrix, argmax_spans = pcfg._viterbi(*params)

      kl = torch.zeros_like(nll) if kl is None else kl
      total_nll += nll.sum().item()
      total_kl  += kl.sum().item()
      num_sents += batch_size
      num_words += batch_size*(length +1) # we implicitly generate </s> so we explicitly count it
      for b in range(batch_size):
        span_b = [(a[0], a[1]) for a in argmax_spans[b] if a[0] != a[1]] #ignore labels
        span_b_set = set(span_b[:-1])        
        gold_b_set = set(gold_spans[b][:-1])
        tp, fp, fn = get_stats(span_b_set, gold_b_set) 
        corpus_f1[0] += tp
        corpus_f1[1] += fp
        corpus_f1[2] += fn
        # sent-level F1 is based on L83-89 from https://github.com/yikangshen/PRPN/test_phrase_grammar.py

        model_out = span_b_set
        std_out = gold_b_set
        overlap = model_out.intersection(std_out)
        prec = float(len(overlap)) / (len(model_out) + 1e-8)
        reca = float(len(overlap)) / (len(std_out) + 1e-8)
        if len(std_out) == 0:
          reca = 1. 
          if len(model_out) == 0:
            prec = 1.
        f1 = 2 * prec * reca / (prec + reca + 1e-8)
        sent_f1.append(f1)
      #break
  tp, fp, fn = corpus_f1  
  prec = tp / (tp + fp)
  recall = tp / (tp + fn)
  #prec = (tp + 1e-6) / (tp + fp + 2e-6)
  #recall = (tp + 1e-6) / (tp + fn + 2e-6)
  corpus_f1 = 2*prec*recall/(prec+recall) if prec+recall > 0 else 0.
  sent_f1 = np.mean(np.array(sent_f1))
  recon_ppl = np.exp(total_nll / num_words)
  ppl_elbo = np.exp((total_nll + total_kl)/num_words) 
  kl = total_kl /num_sents
  print('ReconPPL: %.2f, KL: %.4f, PPL (Upper Bound): %.2f' %
        (recon_ppl, kl, ppl_elbo))
  print('Corpus F1: %.2f, Sentence F1: %.2f' %
        (corpus_f1*100, sent_f1*100))
  model.train()
  return ppl_elbo, sent_f1*100

if __name__ == '__main__':
  args = parser.parse_args()
  if os.path.exists(args.save_path):
      print(f'Warning: the folder {args.save_path} exists.')
  else:
      print('Creating {}'.format(args.save_path))
      os.mkdir(args.save_path)
  logger = logging.getLogger(__name__)
  logger.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
  handler = logging.FileHandler(os.path.join(args.save_path, 'train.log'), 'w')
  handler.setLevel(logging.INFO)
  handler.setFormatter(formatter)
  logger.addHandler(handler)
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  console.setFormatter(formatter)
  logger.addHandler(console)
  logger.propagate = False
  logger.info('cuda:{}@{}'.format(args.gpu, os.uname().nodename))
  logger.info(args)
  main(args, logger.info)
