from torch.utils.checkpoint import checkpoint as ckp

MAXVAL = 1e9

def checkpoint(fn):
    def wrapper(*args, **kwargs):
        return ckp(fn, *args, **kwargs)
    return wrapper

def stripe(x, n, w, offset=(0, 0), dim=1):
    assert dim in {0, 1}, f"strip along row (1) or column (0)."
    x = x.contiguous()
    b, N = x.shape[:2]
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (N + 1) * numel
    stride[2] = (1 if dim == 1 else N) * numel
    offset = (offset[0] * N + offset[1]) * numel
    size = (b, n, w) if x.dim() <= 3 else (b, n, w, *list(x.shape[3:]))
    return x.as_strided(
        size=size, stride=stride, storage_offset=offset
    )

def striped_copy_(x, y, w):
    x = x.contiguous()
    b, N, N = x.shape[:3]
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (N + 1) * numel
    offset = w * numel 
    if x.dim() <= 3:
        size = (b, N - w) 
        stride = stride[:2]
    else:
        stride = stride[:2] + stride[3:]
        size = (b, N - w, *list(x.shape[3:]))
    x.as_strided(
        size=size, stride=stride, storage_offset=offset
    ).copy_(y)

def striped_copy(x, w):
    x = x.contiguous()
    b, N, N = x.shape[:3]
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (N + 1) * numel
    offset = w * numel 
    if x.dim() <= 3:
        size = (b, N - w) 
        stride = stride[:2]
    else:
        stride = stride[:2] + stride[3:]
        size = (b, N - w, *list(x.shape[3:]))
    return x.as_strided(
        size=size, stride=stride, storage_offset=offset
    )
