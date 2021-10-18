import torch

from . import SentenceCFG, MAXVAL
from . import striped_copy_, stripe, checkpoint

class TDSeqCFG(SentenceCFG):
    
    @torch.enable_grad()
    def _dp(self, params, lengths, *args, cky=False, mbr=False, require_marginal=False, **kwargs):
        # mbr decoding only
        terms, (p_rules, l_rules, r_rules), roots = params

        device = roots.device
        b, N, T = terms.shape
        r = p_rules.shape[-1]
        NT = roots.shape[-1]
        S = NT + T

        #@checkpoint
        def transform_rank(x, y):
            return (
                x.unsqueeze(-1) + y.unsqueeze(1) # (b, N, T, 1) + (b, 1, T, r)
            ).logsumexp(2) # (b, N, r) 

        #@checkpoint
        def merge_children(x, y):
            c_r = (x + y).logsumexp(2, keepdim=True) # (b, kw, w, r) + (b, kw, w, r)
            return (c_r + p_rules.unsqueeze(1)).logsumexp(-1) # (b, kw, 1, r) + (b, 1, NT, r) 

        l_term = l_rules[:, NT:].contiguous()
        l_term = transform_rank(terms, l_term) 
        r_term = r_rules[:, NT:].contiguous()
        r_term = transform_rank(terms, r_term)

        l_nonterm = l_rules[:, :NT].contiguous()
        r_nonterm = r_rules[:, :NT].contiguous()

        N += 1
        beta = torch.zeros(b, N, N, NT, device=device).fill_(-MAXVAL)
        l_beta = torch.zeros(b, N, N, r, device=device).fill_(-MAXVAL) 
        r_beta = torch.zeros(b, N, N, r, device=device).fill_(-MAXVAL) 
        ntype = NT if require_marginal else 1 # mem
        typed_spans = torch.zeros(
            b, N, N, ntype, device=device
        ).requires_grad_(mbr or require_marginal)

        striped_copy_(l_beta, l_term, 1)
        striped_copy_(r_beta, r_term, 1)

        for w in range(2, N):
            indice = torch.arange(N - w, device=device)
            Y = stripe(l_beta, N - w, w - 1, (0, 1), 1)
            Z = stripe(r_beta, N - w, w - 1, (1, w), 0)
            X = merge_children(Y.clone(), Z.clone()) # (b, kw, NT)
            X = X + typed_spans[:, indice, indice + w]
            striped_copy_(beta, X, w)
            if w + 1 < N: # cache
                l_X = transform_rank(X, l_nonterm)
                striped_copy_(l_beta, l_X, w)
                r_X = transform_rank(X, r_nonterm)
                striped_copy_(r_beta, r_X, w)
        indice = torch.arange(b, device=device)
        final = beta[indice, 0, lengths] + roots 
        ll = final.logsumexp(-1)
        
        argmax = (None if not mbr else 
            self._extract_parses(ll, typed_spans, lengths, mbr=mbr)
        )
        if not require_marginal:
            return ll, argmax

        marginal = self._compute_marginal(ll, typed_spans, lengths)
        return ll, argmax, marginal
