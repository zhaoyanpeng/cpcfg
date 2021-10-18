import torch
import warnings

def get_stats(span1, span2):
    tp = fp = fn = 0
    for span in span1:
        if span in span2:
            tp += 1
        else:
            fp += 1
    for span in span2:
        if span not in span1:
            fn += 1
    return tp, fp, fn

def update_stats(pred_span, gold_spans, stats):
    for gold_span, stat in zip(gold_spans, stats):
        tp, fp, fn = get_stats(pred_span, gold_span)
        stat[0] += tp
        stat[1] += fp
        stat[2] += fn

def get_f1s(stats):
    f1s = []
    for stat in stats:
        p = stat[0] / (stat[0] + stat[1]) if stat[0] + stat[1] > 0 else 0.
        r = stat[0] / (stat[0] + stat[2]) if stat[0] + stat[2] > 0 else 0.
        f1 = 2 * p * r / (p + r) * 100 if p + r > 0 else 0.
        f1s.append(f1)
    return f1s

def numel(model: torch.nn.Module, trainable: bool = False):
    parameters = list(model.parameters())
    if trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique) 

def detect_nan(x):
    return torch.isnan(x).any(), torch.isinf(x).any()

def postprocess_parses(spans, lengths):
    argmax_spans = list()
    argmax_trees = list()
    for i, span_list in enumerate(spans):
        span_list = [(l, r - 1, 0) for l, r in span_list]
        span_list.sort(key=lambda x: x[1] - x[0])
        argmax_spans.append(span_list)

        tree = {i: str(i) for i in range(lengths[i])}
        for l, r, _ in span_list:
            if l == r:
                continue
            span = '({} {})'.format(tree[l], tree[r])
            tree[r] = tree[l] = span
        argmax_trees.append(tree[0])
    return argmax_spans, argmax_trees

def extract_parse(span, length, inc=1):
    tree = [(i, str(i)) for i in range(length)]
    tree = dict(tree)
    spans = []
    N = span.shape[0]
    cover = span.nonzero()
    #assert cover.shape[0] == N * 2 - 1, \
    #    f"Invalid parses: {length} spans at level 0:\n{span[0]} {cover.shape} != {N * 2 - 1}"
    try:
        fake_me = False
        for i in range(cover.shape[0]):
            if i >= N * 2 - 1: break
            w, r, A = cover[i].tolist()
            w = w + inc 
            r = r + w 
            l = r - w 
            spans.append((l, r, A))
            if l != r:
                span = '({} {})'.format(tree[l], tree[r])
                tree[r] = tree[l] = span
    except Exception as e:
        fake_me = True 
        warnings.warn(f"unparsable because `{e}`.")
    if fake_me or cover.shape[0] > N * 2 - 1:
        spans = [(l, length - 1, 0) for l in range(0, length -1)] 
        tree = dict([(i, str(i)) for i in range(length)])
        spans.reverse()
        for l, r, _ in spans:
            tree[r] = tree[l] = '({} {})'.format(tree[l], tree[r])
    return spans, tree[0]

def extract_parses(matrix, lengths, kbest=False, inc=1):
    batch = matrix.shape[1] if kbest else matrix.shape[0]
    spans = []
    trees = []
    for b in range(batch):
        if kbest:
            span, tree = extract_parses(
                matrix[:, b], [lengths[b]] * matrix.shape[0], kbest=False, inc=inc
            ) 
        else:
            span, tree = extract_parse(matrix[b], lengths[b], inc=inc)
        trees.append(tree)
        spans.append(span)
    return spans, trees 

def build_parse(spans, caption):
    tree = [[i, word, 0, 0] for i, word in enumerate(caption)]
    for l, r in spans:
        if l != r:
            tree[l][2] += 1
            tree[r][3] += 1
    new_tree = ["".join(["("] * nl) + word + "".join([")"] * nr) for i, word, nl, nr in tree] 
    return " ".join(new_tree)

def get_tree(actions, sent=None, SHIFT=0, REDUCE=1):
    #input action and sent (lists), e.g. S S R S S R R, A B C D
    #output tree ((A B) (C D))
    stack = []
    pointer = 0
    if sent is None:
        sent = list(map(str, range((len(actions) + 1) // 2)))
    #assert(len(actions) == 2 * len(sent) - 1)
    for action in actions:
        if action == SHIFT:
            word = sent[pointer]
            stack.append(word)
            pointer += 1
        elif action == REDUCE:
            right = stack.pop()
            left = stack.pop()
            stack.append('(' + left + ' ' + right + ')')
    assert(len(stack) == 1)
    return stack[-1]

def get_actions(tree, SHIFT=0, REDUCE=1, OPEN='(', CLOSE=')'):
    #input tree in bracket form: ((A B) (C D))
    #output action sequence: S S R S S R R
    actions = []
    tree = tree.strip()
    i = 0
    num_shift = 0
    num_reduce = 0
    left = 0
    right = 0
    while i < len(tree):
        if tree[i] != ' ' and tree[i] != OPEN and tree[i] != CLOSE: #terminal      
            if tree[i-1] == OPEN or tree[i-1] == ' ':
                actions.append(SHIFT)
                num_shift += 1
        elif tree[i] == CLOSE:
            actions.append(REDUCE)
            num_reduce += 1
            right += 1
        elif tree[i] == OPEN:
            left += 1
        i += 1
    assert(num_shift == num_reduce + 1)
    return actions

