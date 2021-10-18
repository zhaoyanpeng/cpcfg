import torch

from . import MAXVAL
from . import striped_copy_, striped_copy, stripe, checkpoint

class SentenceCFG:

    def __init__(self):
        pass

    def partition(self, params, lengths, *args, cky=False, mbr=False, require_marginal=False, **kwargs):
        return self._dp(params, lengths, cky=cky, mbr=mbr, require_marginal=require_marginal)

    @torch.enable_grad()
    def argmax(self, params, lengths, *args, cky=False, mbr=False, **kwargs):
        assert cky or mbr, "please specify the decoding method (cky | mbr)."
        return self._dp(params, lengths, cky=cky, mbr=mbr)

    def _dp(self, params, lengths, *args, cky=False, mbr=False, **kwargs):
        pass 

    def _compute_marginal(self, ll, spans, lengths):
        # span marginals by type
        marginals = torch.autograd.grad(
            ll.sum(), spans, create_graph=True, only_inputs=True, allow_unused=False
        )[0]
        # linearization
        beg = 0
        b, N, _, NT = marginals.shape # (b, N, N, NT)
        scores = torch.zeros(b, int((N - 2) * (N - 1) / 2), NT, device=marginals.device)
        for w in range(2, N):
            w_spans = striped_copy(marginals, w)
            kw = w_spans.shape[1] # (b, kw, NT)
            scores[:, beg : beg + kw] = w_spans
            beg = beg + kw
        return scores # marginals #

    def _extract_parses(self, ll, spans, lengths, mbr=False):
        b, N = spans.shape[:2] # (b, N, N)
        if N < 3: # trivial spans for len = 2
            return [[[0, 1], [1, 2], [0, 2]] for _ in range(b)]
        #ll.sum().backward(retain_graph=True) # gradients would be accumulated
        #marginals = spans.grad
        marginals = torch.autograd.grad(
            ll.sum(), spans, create_graph=True, only_inputs=True, allow_unused=False
        )[0]
        if spans.dim() == 4: # unlabeled
            marginals = marginals.sum(-1)
        if mbr: # 
            return self._mbr(marginals.detach(), lengths)
        parses = [[] for _ in range(b)]
        spans = marginals.nonzero().tolist()
        for span in spans: # left boundary should minus 1
            parses[span[0]].append((span[1], span[2])) 
        return parses
    
    @torch.no_grad()
    def _mbr(self, spans, lengths):
        b, N = spans.shape[:2]
        beta = torch.zeros_like(spans).fill_(-MAXVAL)
        best = torch.zeros_like(spans).long()
        striped_copy_(beta, striped_copy(spans, 1), 1)
        for w in range(2, N):
            indice = torch.arange(N - w, device=spans.device)
            Y = stripe(beta, N - w, w - 1, (0, 1), 1)
            Z = stripe(beta, N - w, w - 1, (1, w), 0)
            X, k = (Y + Z).max(2) # (b, kw, w - 1)
            X = X + striped_copy(spans, w)
            striped_copy_(beta, X, w)
            striped_copy_(best, indice.unsqueeze(0) + k + 1, w)

        def backtrack(best, i, j):
            # j should minus 1 so as to be consistent with gold tree repr.
            if i + 1 == j:
                return [(i, j)]
            k = best[i][j]
            l_tree = backtrack(best, i, k)
            r_tree = backtrack(best, k, j)
            return l_tree + r_tree + [(i, j)]

        parses = [
            backtrack(best[i].tolist(), 0, l) for i, l in enumerate(lengths.tolist())
        ]
        return parses
