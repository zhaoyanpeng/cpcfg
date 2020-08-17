import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ResLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.linear(x) + x

class CPCFG(torch.nn.Module):
    def __init__(self, V, NT, T, *args, 
                 h_dim = 512,
                 w_dim = 512,
                 z_dim = 64,
                 s_dim = 256, **kwargs): 
        super(CPCFG, self).__init__()
        assert z_dim >= 0
        self.NT_T = NT + T
        self.NT = NT
        self.T = T
        self.z_dim = z_dim
        self.s_dim = s_dim

        self.share_term = kwargs.get("share_term", False)
        self.share_rule = kwargs.get("share_rule", False)
        self.share_root = kwargs.get("share_root", False)
        self.wo_enc_emb = kwargs.get("wo_enc_emb", False)

        self.term_emb = nn.Parameter(torch.randn(T, s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(NT, s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, s_dim))

        rule_dim = s_dim if self.share_rule else s_dim + z_dim
        self.rule_mlp = nn.Linear(rule_dim, self.NT_T ** 2)
        root_dim = s_dim if self.share_root else s_dim + z_dim
        self.root_mlp = nn.Sequential(nn.Linear(root_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, NT))
        if z_dim > 0:
            i_dim = w_dim
            self.enc_emb = lambda x: x 
            if not self.wo_enc_emb:
                self.enc_emb = nn.Embedding(V, w_dim)
                self.enc_rnn = nn.LSTM(w_dim, h_dim, 
                    bidirectional=True, num_layers=1, batch_first=True)
                i_dim = h_dim * 2
            self.enc_out = nn.Linear(i_dim, z_dim * 2)

        term_dim = s_dim if self.share_term else s_dim + z_dim
        self.term_mlp = nn.Sequential(nn.Linear(term_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, V))
        self._initialize()

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def update_state_dict(self, new_state, strict=True):
        self.load_state_dict(new_state, strict=strict) 

    def kl(self, mean, lvar):
        return -0.5 * (lvar - torch.pow(mean, 2) - torch.exp(lvar) + 1)

    def enc(self, x, lengths, max_pooling=True, enforce_sorted=False):
        x_embbed = self.enc_emb(x)
        x_packed = pack_padded_sequence(
            x_embbed, lengths, batch_first=True, enforce_sorted=enforce_sorted
        )
        h_packed, _ = self.enc_rnn(x_packed)
        if max_pooling:
            padding_value = float("-inf")
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.max(1)[0]
        else:
            padding_value = 0
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.sum(1)[0] / lengths.unsqueze(-1)
        out = self.enc_out(h)
        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim :]
        return mean, lvar

    def forward(self, x, lengths, *args, txt=None, txt_lengths=None, use_mean=False, **kwargs):
        """ x, lengths: word ids; txt, txt_lengths: sub-word ids """
        b, n = x.shape[:2]
        batch_size = b 
        if self.z_dim > 0:
            max_pooling = kwargs.get("max_pooling", True)
            enforce_sorted = kwargs.get("enforce_sorted", False)
            item = (x, lengths, txt, txt_lengths) + args if txt is not None else x
            mean, lvar = self.enc(
                item, lengths, max_pooling=max_pooling, enforce_sorted=enforce_sorted
            )
            z = mean
            if not use_mean:
                z = mean.new(b, mean.size(1)).normal_(0, 1)
                z = (0.5 * lvar).exp() * z + mean
            kl = self.kl(mean, lvar).sum(1) 
        else:
            z = torch.zeros(b, 1).cuda()
            kl = None
        self.z = z

        def roots():
            root_emb = self.root_emb.expand(b, self.s_dim)
            if self.z_dim > 0 and not self.share_root:
                root_emb = torch.cat([root_emb, self.z], -1)
            root_prob = F.log_softmax(self.root_mlp(root_emb), -1)
            return root_prob
        
        def terms():
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.T, self.s_dim
            ) 
            if self.z_dim > 0 and not self.share_term:
                #z_expand = self.z.unsqueeze(1).unsqueeze(2).expand(
                #    b, n, self.T, self.z_dim
                #) # it indeed makes a difference, weird.
                z_expand = z.unsqueeze(1).expand(b, n, self.z_dim)
                z_expand = z_expand.unsqueeze(2).expand(b, n, self.T, self.z_dim)
                term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = F.log_softmax(self.term_mlp(term_emb), -1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules():
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
                b, self.NT, self.s_dim
            )
            if self.z_dim > 0 and not self.share_rule:
                z_expand = self.z.unsqueeze(1).expand(
                    b, self.NT, self.z_dim
                )
                nonterm_emb = torch.cat([nonterm_emb, z_expand], -1)
            rule_prob = F.log_softmax(self.rule_mlp(nonterm_emb), -1)
            rule_prob = rule_prob.view(b, self.NT, self.NT_T, self.NT_T)
            return rule_prob

        roots_ll, terms_ll, rules_ll = roots(), terms(), rules()
        return (terms_ll, rules_ll, roots_ll), kl 

