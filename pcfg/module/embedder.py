import torch
from torch.nn import CrossEntropyLoss

from gensim.test.utils import datapath
from gensim.models import KeyedVectors

class PretrainedEncoder(torch.nn.Module):
    def __init__(self, model_name, out_dim=-1, as_encoder=False, fine_tuned=False, pooler_type="max"):
        super(PretrainedEncoder, self).__init__()
        self.model_name = model_name
        self.as_encoder = as_encoder
        self.fine_tuned = fine_tuned
        self.pooler_type = pooler_type
        if "xlm" in self.model_name:
            from transformers import XLMTokenizer
            from .transformer import XLMWithLMHeadModel as Xlm 
            self.tokenizer = XLMTokenizer.from_pretrained(self.model_name)
            self.model = Xlm.from_pretrained(self.model_name) 
        elif "bert" in self.model_name:
            from transformers import BertTokenizer
            from . import BertForMaskedLM as Bert 
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = Bert.from_pretrained(self.model_name) 
        elif "xlnet" in self.model_name:
            from transformers import XLNetTokenizer
            from .transformer import XLNetLMHeadModel as Xlnet 
            self.tokenizer = XLNetTokenizer.from_pretrained(self.model_name)
            self.model = Xlnet.from_pretrained(self.model_name)
        else:
            self.tokenizer = None 
            self.model = None
            raise TypeError("Support only bert & xlnet models.")
        in_dim = self.model.get_input_embeddings().weight.shape[-1] 
        self.linear = (
            torch.nn.Linear(in_dim, out_dim, bias=False)
            if out_dim > 0 else torch.nn.Identity() #None #lambda x: x
        )
        self._output_dim = out_dim if out_dim > 0 else in_dim

    @property
    def output_dim(self):
        return self._output_dim
    
    def forward(self, X):
        _, _, sub_words, sub_indexes = X 
        sub_word_mask = sub_words.ne(self.tokenizer.pad_token_id) #.float() 
        # https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
        # https://github.com/huggingface/transformers/issues/1225#issuecomment-529516396
        with torch.set_grad_enabled(self.fine_tuned):
            if not self.fine_tuned: 
                self.model.eval()
            if "xlm" in self.model_name:
                masked_lm_loss, word_vect, hidden = self.xlm_encoder(
                    sub_words, sub_word_mask, sub_indexes
                ) 
            elif "bert" in self.model_name:
                masked_lm_loss, word_vect, hidden = self.bert_encoder(
                    sub_words, sub_word_mask, sub_indexes
                ) 
            elif "xlnet" in self.model_name:
                masked_lm_loss, word_vect, hidden = self.xlnet_encoder(
                    sub_words, sub_word_mask, sub_indexes
                ) 
            else:
                raise TypeError("Unsupported LMs {}".format(self.model_name))
        word_vect = self.linear(word_vect)
        return masked_lm_loss, word_vect, hidden 

    def xlm_encoder(self, tokens, attention_mask, word_indice):
        if self.as_encoder: # sequence encoder
            labels = tokens.masked_fill(
                tokens == self.tokenizer.pad_token_id, -100
            ) 
            loss, _, hidden, seq_embed = self.model(
                tokens, attention_mask=attention_mask, labels=labels, 
                word_indice=word_indice, pooler_type=self.pooler_type,
            )
            return loss, hidden, seq_embed
        return None, None, None

    def bert_encoder(self, tokens, attention_mask, word_indice):
        if self.as_encoder: # sequence encoder
            labels = tokens.masked_fill(
                tokens == self.tokenizer.pad_token_id, -100
            ) 
            loss, _, hidden, seq_embed = self.model(
                tokens, attention_mask=attention_mask, labels=labels,
                word_indice=word_indice, pooler_type=self.pooler_type,
            )
            return loss, hidden, seq_embed

        logits, hidden = list(), list()
        for i in range(tokens.size(-1)):
            masked_tokens = tokens.clone().detach()
            masked_tokens[:, i] = self.tokenizer.mask_token_id 
            i_logits, i_hidden, _ = self.model(
                masked_tokens, attention_mask=attention_mask
            )
            logits.append(i_logits[:, i])
            hidden.append(i_hidden[:, i])
        logits = torch.stack(logits, 1)
        hidden = torch.stack(hidden, 1)

        loss_fct = CrossEntropyLoss()  # -100 index = padding token
        masked_lm_loss = loss_fct(
            logits.view(-1, self.model.config.vocab_size), 
            tokens.masked_fill(tokens == self.tokenizer.pad_token_id, -100).view(-1)
        )

        word_vect = torch.gather( # for word-level LMs and latent z
            hidden, 1, word_indice.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        )
        return masked_lm_loss, word_vect, None 

    def xlnet_encoder(self, tokens, attention_mask, word_indice):
        if self.as_encoder: # sequence encoder
            labels = tokens.masked_fill(
                tokens == self.tokenizer.pad_token_id, -100
            ) 
            loss, _, hidden, seq_embed = self.model(
                tokens, attention_mask=attention_mask, labels=labels, 
                word_indice=word_indice, pooler_type=self.pooler_type,
            )
            return loss, hidden, seq_embed
        return None, None, None

    def __repr__(self):
        def _addindent(s_, numSpaces):
            s = s_.split('\n')
            # don't do anything for single-line stuff
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(numSpaces * ' ') + line for line in s]
            s = '\n'.join(s)
            s = first + '\n' + s
            return s

        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            if key == "model":
                mod_str = "PretrainedLM({}, {}, grad={})".format(
                    self.model_name, tuple([self.output_dim]), self.fine_tuned
                ) 
            else:
                mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

class PartiallyFixedEmbedding(torch.nn.Module):
    def __init__(self, vocab, w2vec_file, word_dim=-1, out_dim=-1):
        super(PartiallyFixedEmbedding, self).__init__()
        nword = len(vocab)
        model = KeyedVectors.load_word2vec_format(datapath(w2vec_file), binary=False)
        masks = [1 if vocab.idx2word[k] in model.vocab else 0 for k in range(nword)]
        idx2fixed = [k for k in range(nword) if masks[k]]
        idx2tuned = [k for k in range(nword) if not masks[k]]
        arranged_idx = idx2fixed + idx2tuned
        idx_mapping = {idx: real_idx for real_idx, idx in enumerate(arranged_idx)}
        self.register_buffer("realid", torch.tensor(
            [idx_mapping[k] for k in range(nword)], dtype=torch.int64
        ))
        self.idx_mapping = idx_mapping
        self.n_fixed = sum(masks)
        n_tuned = nword - self.n_fixed

        weight = torch.empty(nword, model.vector_size)
        for k, idx in vocab.word2idx.items():
            real_idx = idx_mapping[idx]
            if k in model.vocab:
                weight[real_idx] = torch.tensor(model[k])

        self.tuned_weight = torch.nn.Parameter(torch.empty(n_tuned, model.vector_size)) 
        torch.nn.init.kaiming_uniform_(self.tuned_weight)
        weight[self.n_fixed:] = self.tuned_weight
        self.register_buffer("weight", weight)
         
        add_dim = word_dim - model.vector_size if word_dim > model.vector_size else 0 
        self.tuned_vector = torch.nn.Parameter(torch.empty(nword, add_dim))
        if add_dim > 0: 
            torch.nn.init.kaiming_uniform_(self.tuned_vector)
        in_dim = model.vector_size if add_dim == 0 else word_dim 

        self.linear = (
            torch.nn.Linear(in_dim, out_dim, bias=False)
            if out_dim > 0 else torch.nn.Identity() #None #lambda x: x
        )
        self._output_dim = out_dim if out_dim > 0 else in_dim
        del model

    @property
    def output_dim(self):
        return self._output_dim

    def __setstate__(self, state):
        super(PartiallyFixedEmbedding, self).__setstate__(state)
        pass

    def extra_repr(self):
        mod_keys = self._modules.keys()
        all_keys = self._parameters.keys()
        extra_keys = all_keys - mod_keys
        extra_keys = [k for k in all_keys if k in extra_keys]
        extra_lines = []
        for key in extra_keys:
            attr = getattr(self, key)
            if not isinstance(attr, torch.nn.Parameter):
                continue
            extra_lines.append("({}): Tensor{}".format(key, tuple(attr.size())))
        return "\n".join(extra_lines)

    def reindex(self, X):
        return X.clone().cpu().apply_(self.idx_mapping.get)

    def get_tunable_word_list(self):
        nword = self.weight.shape[0]
        idx_mapping = {v: k for k, v in self.idx_mapping.items()}
        return [idx_mapping[i] for i in range(self.n_fixed, nword)]

    def get_word_vector(self, wid):
        realid = self.realid[wid]
        if realid == 0: return self.weight[realid] # id of the unknown word
        assert realid >= self.n_fixed
        realid = realid - self.n_fixed
        return self.tuned_weight[realid]

    def set_word_vector(self, wid, vector):
        realid = self.realid[wid]
        assert realid >= self.n_fixed
        realid = realid - self.n_fixed
        with torch.no_grad():
            self.tuned_weight[realid] = vector

    def bmm(self, X):
        self.realid.detach_()
        self.weight.detach_()
        self.weight[self.n_fixed:] = self.tuned_weight
        word_emb = torch.cat([self.weight, self.tuned_vector], -1)
        word_emb = word_emb[self.realid] 
        word_emb = self.linear(word_emb)
        x_shape = X.size()
        w_logit = torch.matmul(
            X.view(-1, x_shape[-1]), word_emb.transpose(0, 1)
        )
        w_logit = w_logit.view(x_shape[:-1] + (w_logit.size(-1),))
        return w_logit 

    def forward(self, X):
        if X.dtype != torch.int64:  
            return self.bmm(X) # w/o linear 
        self.weight.detach_()
        self.weight[self.n_fixed:] = self.tuned_weight
        weight = torch.cat([self.weight, self.tuned_vector], -1)
        #X = X.clone().cpu().apply_(self.idx_mapping.get) # only work on cpus
        X = X.clone().cpu().apply_(self.idx_mapping.get).cuda() 
        #print(X, weight.device, self.weight.device, self.tuned_vector.device)
        word_vect = torch.nn.functional.embedding(X, weight, None, None, 2.0, False, False)
        if self.linear is not None:
            word_vect = self.linear(word_vect)
        return word_vect

