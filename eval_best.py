import os, re, sys

import argparse
import pickle, json, copy, time
import numpy as np
from collections import defaultdict

import torch
from torch import cuda

from torch_struct import SentCFG
from torch_struct.networks import CPCFG, RoughCCFG, CompoundCFG

from PCFG import PCFG
from utils import get_stats, get_nonbinary_spans, extract_parses
from eval import clean_number, get_tags_tokens_lowercase, get_actions


def build_parse(spans, caption):
    tree = [[i, word, 0, 0] for i, word in enumerate(caption)]
    for l, r in spans:
        if l != r:
            tree[l][2] += 1
            tree[r][3] += 1
    new_tree = ["".join(["("] * nl) + word + "".join([")"] * nr) for i, word, nl, nr in tree] 
    return " ".join(new_tree)

parser = argparse.ArgumentParser()

# Data path options
parser.add_argument('--data_file', default='data/ptb-test.txt')
parser.add_argument('--model_file', default='')
parser.add_argument('--out_file', default='pred-parse.txt')
parser.add_argument('--gold_out_file', default='gold-parse.txt')
parser.add_argument('--data_path', type=str, default='./')
parser.add_argument('--vocab_name', type=str, default='vocab')
parser.add_argument('--infer_fast', default=True, type=bool, help='model name (1st/2nd)')
# Inference options
parser.add_argument('--use_mean', default=1, type=int, help='use mean from q if = 1')
parser.add_argument('--kbest', default=1, type=int, help='k-best parsing')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--inc', default=1, type=int, help='0 if with postag')
parser.add_argument('--kim', default=True, type=bool, help='0 if with what')

def main(args):
    sig = args.data_file.split('.')[-1]
    pkl = sig == 'pkl'
    print('loading model from ' + args.model_file)
    torch.nn.Module.dump_patches = True
    checkpoint = torch.load(args.model_file, map_location='cpu')
    model = checkpoint['model']
    #cuda.set_device(args.gpu)
    print('--------------------------------')
    print('Checking validation perf...')    
    if args.kim:
      kim_inf=False
      kim_test(model, checkpoint, args, kim_inf=kim_inf)
    else:
      raise NameError("Invalid eval choice: {}".format(args.kim)) 
    print('--------------------------------')

def kim_test(model, checkpoint, args, kim_inf=False):
    copy = True
    opt = checkpoint['opt'] if "opt" in checkpoint else checkpoint['args'] 
    if isinstance(opt, dict):
        class Namespace:
            pass
        opt_new = Namespace() 
        for k, v in opt.items():
            setattr(opt_new, k, v)
        setattr(opt_new, "vocab_size", len(checkpoint['word2idx']))
        opt = opt_new
        copy = False
    
    pcfg = PCFG(opt.nt_states, opt.t_states)

    if True:
        model = make_model_kim(model, opt)
    word2idx = checkpoint['word2idx'] # dependent on `copy`

    model.eval()
    model.cuda()
    eval_ptb_fast(word2idx, model, pcfg, kim_inf=kim_inf)

def make_model_kim(best_model, args):
    model_type = args.model_type
    if model_type == '4th':
        model = CPCFG
    else:
        raise NameError("Invalid parser type: {}".format(opt.parser_type)) 
    kwarg = {
        "share_term": getattr(args, "share_term", False),
        "share_rule": getattr(args, "share_rule", False),
        "share_root": getattr(args, "share_root", False),
        "wo_enc_emb": getattr(args, "wo_enc_emb", False),
    }
    model = model(args.vocab_size, args.nt_states, args.t_states, 
                  h_dim = args.h_dim,
                  w_dim = args.w_dim,
                  z_dim = args.z_dim,
                  s_dim = args.state_dim, **kwarg)
    model.load_state_dict(best_model.state_dict(), strict=False)
    return model

def eval_ptb_fast(word2idx, model, pcfg, inc=0, kim_inf=False):
  #print('loading model from ' + args.model_file)
  #checkpoint = torch.load(args.model_file)
  #model = checkpoint['model']
  #cuda.set_device(args.gpu)
  #model.eval()
  #model.cuda()
  total_kl = 0.
  total_nll = 0.
  num_sents = 0
  num_words = 0
  #word2idx = checkpoint['word2idx']
  corpus_f1 = [0., 0., 0.] 
  sent_f1 = [] 
  pred_out = open(args.out_file, "w")
  gold_out = open(args.gold_out_file, "w")
  with torch.no_grad():
    for tree in open(args.data_file, "r"):
      """
      (caption, spans, labels, tags) = json.loads(tree)
      sent_orig = caption.strip().lower().split()
      sent = [clean_number(w) for w in sent_orig]
      #if len(caption) < min_length or len(caption) > max_length:
      #    continue
      length = len(sent)
      gold_span = spans

      """
      tree = tree.strip()
      action = get_actions(tree)
      tags, sent, sent_lower = get_tags_tokens_lowercase(tree)
      gold_span, binary_actions, nonbinary_actions = get_nonbinary_spans(action)
      length = len(sent)
      sent_orig = sent_lower
      sent = [clean_number(w) for w in sent_orig]
      #"""

      if length == 1:
        continue # we ignore length 1 sents.
      sent_idx = [word2idx[w] if w in word2idx else word2idx["<unk>"] for w in sent]
      sents = torch.from_numpy(np.array(sent_idx)).unsqueeze(0)
      sents = sents.cuda()


      if kim_inf:
        #nll, kl, binary_matrix, argmax_spans = model(sents, argmax=True, use_mean=(args.use_mean==1))
        lengths = torch.tensor([length], device=sents.device).long() 
        params, kl = model(sents, lengths, use_mean=(args.use_mean==1))
        log_Z = pcfg._inside(*params)
        nll = -log_Z
        with torch.no_grad():
          max_score, binary_matrix, argmax_spans = pcfg._viterbi(*params)
          model.tags = pcfg.argmax_tags
      else:
        lengths = torch.tensor([length], device=sents.device).long() 
        params, kl = model(sents, lengths, use_mean=(args.use_mean==1))

        dist = SentCFG(params, lengths=lengths)
        spans = dist.argmax[-1]
        argmax_spans, _ = extract_parses(spans, lengths.tolist(), inc=0)
        nll = -dist.partition

        tree = build_parse(gold_span, sent_orig)

        binary_matrix = torch.zeros(1, length, length)
        argmax_tags = torch.zeros(1, length)
        for l, r, A in argmax_spans[0]:
            binary_matrix[0][l][r] = 1
            if l == r:
              argmax_tags[0][l] = A 
        model.tags = argmax_tags


      total_nll += nll.sum().item() if nll is not None else 0
      total_kl  += kl.sum().item() if kl is not None else 0
      num_sents += 1
      # the grammar implicitly generates </s> token, in contrast to a sequential lm which must explicitly
      # generate it. the sequential lm takes into account </s> token in perplexity calculations, so
      # for comparison the pcfg must also take into account </s> token, which amounts to just adding
      # one more token to length for each sentence
      num_words += length + 1
      gold_span= [(a[0], a[1]) for a in gold_span]
      pred_span= [(a[0], a[1]) for a in argmax_spans[0] if a[0] != a[1]]
      pred_span_set = set(pred_span[:-1]) #the last span in the list is always the
      gold_span_set = set(gold_span[:-1]) #trival sent-level span so we ignore it
      tp, fp, fn = get_stats(pred_span_set, gold_span_set) 
      corpus_f1[0] += tp
      corpus_f1[1] += fp
      corpus_f1[2] += fn
      # sent-level F1 is based on L83-89 from https://github.com/yikangshen/PRPN/test_phrase_grammar.py

      overlap = pred_span_set.intersection(gold_span_set)
      prec = float(len(overlap)) / (len(pred_span_set) + 1e-8)
      reca = float(len(overlap)) / (len(gold_span_set) + 1e-8)
      if len(gold_span_set) == 0:
        reca = 1. 
        if len(pred_span_set) == 0:
          prec = 1.
      f1 = 2 * prec * reca / (prec + reca + 1e-8)
      sent_f1.append(f1)

      argmax_tags = model.tags[0]
      binary_matrix = binary_matrix[0].cpu().numpy()
      label_matrix = np.zeros((length, length))
      for span in argmax_spans[0]:
        label_matrix[span[0]][span[1]] = span[2]
      pred_tree = {}
      for i in range(length):
        tag = "T-" + str(int(argmax_tags[i].item())+1) 
        pred_tree[i] = "(" + tag + " " + sent_orig[i] + ")"
      for k in np.arange(1, length):
        for s in np.arange(length):
          t = s + k
          if t > length - 1: break
          if binary_matrix[s][t] == 1:
            nt = "NT-" + str(int(label_matrix[s][t])+1)
            span = "(" + nt + " " + pred_tree[s] + " " + pred_tree[t] +  ")"
            pred_tree[s] = span
            pred_tree[t] = span
      pred_tree = pred_tree[0]
      pred_out.write(pred_tree.strip() + "\n")
      gold_out.write(tree.strip() + "\n")
      print(pred_tree)
      if num_sents > 10:
        pass #break
  pred_out.close()
  gold_out.close()
  tp, fp, fn = corpus_f1  
  prec = tp / (tp + fp)
  recall = tp / (tp + fn)
  corpus_f1 = 2*prec*recall/(prec+recall) if prec+recall > 0 else 0.
  sent_f1 = np.mean(np.array(sent_f1))
  recon_ppl = np.exp(total_nll / num_words)
  ppl_elbo = np.exp((total_nll + total_kl)/num_words)
  kl = total_kl /num_sents
  # note that if use_mean == 1, then the PPL upper bound is not a true upper bound
  # run with use_mean == 0, to get the true upper bound
  #print('ReconPPL: %.2f, KL: %.4f, PPL Upper Bound from ELBO: %.2f' %
  #      (recon_ppl, kl, ppl_elbo))
  #print('Corpus F1: %.2f, Sentence F1: %.2f' %
  #      (corpus_f1*100, np.mean(np.array(sent_f1))*100))
  info = '\nReconPPL: {:.2f}, KL: {:.4f}, PPL (Upper Bound): {:.2f}\n' + \
         'Corpus F1: {:.2f}, Sentence F1: {:.2f}'
  info = info.format(
      recon_ppl, kl, ppl_elbo, corpus_f1 * 100, sent_f1 * 100
  )
  print(info)

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
