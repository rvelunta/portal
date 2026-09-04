"""A minimal pure-Python stand-in for the slice of numpy that QM configuration
files use (waveform construction: linspace/arange, exp/sin/cos, elementwise
arithmetic, tolist).

Installed as `numpy` ONLY when the real numpy is not importable -- if numpy is
present it is always preferred. This is not a numpy clone: arrays are nested
Python lists with elementwise operators, there is no broadcasting between
different shapes, and anything unimplemented raises rather than silently
returning something wrong.
"""

import builtins as _bi
import math as _math
import random as _random

pi = _math.pi
e = _math.e
inf = _math.inf
nan = _math.nan
newaxis = None


def _apply1(x, f):
    if isinstance(x, ndarray):
        return ndarray(_apply1(x.data, f))
    if isinstance(x, (list, tuple)):
        return [_apply1(v, f) for v in x]
    return f(x)


def _apply2(a, b, f):
    if isinstance(a, ndarray) or isinstance(b, ndarray):
        return ndarray(_apply2(_raw(a), _raw(b), f))
    a_seq, b_seq = isinstance(a, (list, tuple)), isinstance(b, (list, tuple))
    if a_seq and b_seq:
        if len(a) != len(b):
            raise ValueError(f"shape mismatch: {len(a)} vs {len(b)}")
        return [_apply2(x, y, f) for x, y in zip(a, b)]
    if a_seq:
        return [_apply2(x, b, f) for x in a]
    if b_seq:
        return [_apply2(a, y, f) for y in b]
    return f(a, b)


def _raw(x):
    return x.data if isinstance(x, ndarray) else x


def _flat(x):
    x = _raw(x)
    if isinstance(x, (list, tuple)) or hasattr(x, "__next__"):
        for v in x:
            yield from _flat(v)
    else:
        yield x


class ndarray:
    """Nested-list array with elementwise operators."""

    __array_priority__ = 100

    def __init__(self, data):
        self.data = list(data) if isinstance(data, (list, tuple)) else data

    # ---- container protocol
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for v in self.data:
            yield ndarray(v) if isinstance(v, list) else v

    def __getitem__(self, k):
        if isinstance(k, tuple):
            out = self
            for kk in k:
                out = out[kk]
            return out
        v = self.data[k]
        return ndarray(v) if isinstance(v, list) else v

    def __setitem__(self, k, v):
        self.data[k] = _raw(v)

    def __repr__(self):
        return f"array({self.data!r})"

    def __float__(self):
        vals = list(_flat(self))
        if len(vals) != 1:
            raise TypeError("only size-1 arrays can be converted to a scalar")
        return float(vals[0])

    def __bool__(self):
        vals = list(_flat(self))
        if len(vals) == 1:
            return bool(vals[0])
        return len(vals) > 0 and all(bool(v) for v in vals)

    # ---- attributes
    @property
    def shape(self):
        s, d = [], self.data
        while isinstance(d, list):
            s.append(len(d))
            d = d[0] if d else None
        return tuple(s)

    @property
    def size(self):
        return len(list(_flat(self)))

    @property
    def T(self):
        if len(self.shape) != 2:
            return self
        return ndarray([list(r) for r in zip(*self.data)])

    def tolist(self):
        return self.data

    def copy(self):
        return ndarray(_apply1(self.data, lambda v: v))

    def astype(self, t):
        return ndarray(_apply1(self.data, t))

    def flatten(self):
        return ndarray(list(_flat(self)))

    ravel = flatten

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        flat = list(_flat(self))
        if len(shape) == 1 or -1 in shape and len(shape) == 1:
            return ndarray(flat)
        if len(shape) == 2:
            r, c = shape
            if r == -1:
                r = len(flat) // c
            if c == -1:
                c = len(flat) // r
            return ndarray([flat[i * c:(i + 1) * c] for i in range(r)])
        raise NotImplementedError("reshape beyond 2-D")

    def max(self): return _bi.max(_flat(self))
    def min(self): return _bi.min(_flat(self))
    def sum(self): return _bi.sum(_flat(self))
    def mean(self): return _bi.sum(_flat(self)) / _bi.max(1, self.size)

    def std(self):
        m, n = self.mean(), _bi.max(1, self.size)
        return _math.sqrt(sum((v - m) ** 2 for v in _flat(self)) / n)

    # ---- operators
    def __add__(self, o): return _apply2(self, o, lambda a, b: a + b)
    __radd__ = __add__
    def __sub__(self, o): return _apply2(self, o, lambda a, b: a - b)
    def __rsub__(self, o): return _apply2(o, self, lambda a, b: a - b)
    def __mul__(self, o): return _apply2(self, o, lambda a, b: a * b)
    __rmul__ = __mul__
    def __truediv__(self, o): return _apply2(self, o, lambda a, b: a / b)
    def __rtruediv__(self, o): return _apply2(o, self, lambda a, b: a / b)
    def __floordiv__(self, o): return _apply2(self, o, lambda a, b: a // b)
    def __mod__(self, o): return _apply2(self, o, lambda a, b: a % b)
    def __pow__(self, o): return _apply2(self, o, lambda a, b: a ** b)
    def __neg__(self): return _apply1(self, lambda v: -v)
    def __abs__(self): return _apply1(self, _bi.abs)
    def __lt__(self, o): return _apply2(self, o, lambda a, b: a < b)
    def __le__(self, o): return _apply2(self, o, lambda a, b: a <= b)
    def __gt__(self, o): return _apply2(self, o, lambda a, b: a > b)
    def __ge__(self, o): return _apply2(self, o, lambda a, b: a >= b)
    def __eq__(self, o): return _apply2(self, o, lambda a, b: a == b)
    def __ne__(self, o): return _apply2(self, o, lambda a, b: a != b)
    __hash__ = None


# ------------------------------------------------------------------ creation

def array(x, dtype=None):
    d = _raw(x)
    d = list(d) if isinstance(d, (list, tuple)) else d
    a = ndarray(_apply1(d, dtype) if dtype else d)
    return a


def asarray(x, dtype=None): return x if isinstance(x, ndarray) else array(x, dtype)


def arange(start, stop=None, step=1, dtype=None):
    if stop is None:
        start, stop = 0, start
    out, v, n = [], start, 0
    while (step > 0 and v < stop - 1e-12) or (step < 0 and v > stop + 1e-12):
        out.append(dtype(v) if dtype else v)
        n += 1
        v = start + n * step
    return ndarray(out)


def linspace(start, stop, num=50, endpoint=True, dtype=None):
    num = int(num)
    if num <= 1:
        return ndarray([start])
    step = (stop - start) / ((num - 1) if endpoint else num)
    out = [start + i * step for i in range(num)]
    return ndarray([dtype(v) for v in out] if dtype else out)


def logspace(start, stop, num=50, base=10.0):
    return ndarray([base ** v for v in linspace(start, stop, num)])


def geomspace(start, stop, num=50):
    return logspace(_math.log10(start), _math.log10(stop), num)


def zeros(n, dtype=float):
    if isinstance(n, (tuple, list)):
        if len(n) == 1:
            return ndarray([0.0] * n[0])
        return ndarray([[0.0] * n[1] for _ in range(n[0])])
    return ndarray([0.0] * int(n))


def ones(n, dtype=float):
    z = zeros(n)
    return _apply2(z, 1.0, lambda a, b: b)


def full(n, v): return _apply2(zeros(n), v, lambda a, b: b)
def zeros_like(a): return _apply1(a, lambda v: 0.0)
def ones_like(a): return _apply1(a, lambda v: 1.0)


def concatenate(arrs, axis=0):
    out = []
    for a in arrs:
        out.extend(_raw(a) if isinstance(_raw(a), list) else [_raw(a)])
    return ndarray(out)


def append(a, v):
    out = list(_raw(a))
    if isinstance(_raw(v), list):
        out.extend(_raw(v))
    else:
        out.append(_raw(v))
    return ndarray(out)


def flip(a): return ndarray(list(reversed(_raw(a))))
def tile(a, n): return ndarray(list(_raw(a)) * int(n))


def repeat(a, n):
    out = []
    for v in _raw(a):
        out.extend([v] * int(n))
    return ndarray(out)


def diff(a):
    d = list(_flat(a))
    return ndarray([d[i + 1] - d[i] for i in range(len(d) - 1)])


def cumsum(a):
    out, s = [], 0
    for v in _flat(a):
        s += v
        out.append(s)
    return ndarray(out)


def where(cond, x, y):
    return _apply2(_apply2(cond, x, lambda c, xv: (c, xv)), y,
                   lambda cx, yv: cx[1] if cx[0] else yv)


def clip(a, lo, hi): return _apply1(a, lambda v: max(lo, min(hi, v)))


# ------------------------------------------------------------------ math ufuncs

def _uf(f):
    return lambda x: _apply1(x, f) if isinstance(x, (ndarray, list, tuple)) else f(x)


exp = _uf(_math.exp)
sin = _uf(_math.sin)
cos = _uf(_math.cos)
tan = _uf(_math.tan)
tanh = _uf(_math.tanh)
sqrt = _uf(_math.sqrt)
log = _uf(_math.log)
log10 = _uf(_math.log10)
log2 = _uf(_math.log2)
floor = _uf(_math.floor)
ceil = _uf(_math.ceil)
sign = _uf(lambda v: (v > 0) - (v < 0))
arctan = _uf(_math.atan)
arcsin = _uf(_math.asin)
arccos = _uf(_math.acos)
degrees = _uf(_math.degrees)
radians = _uf(_math.radians)


def abs(x): return _apply1(x, _bi.abs)                              # noqa: A001
absolute = abs
def round(x, n=0): return _apply1(x, lambda v: _bi.round(v, n))     # noqa: A001
def max(x, *a): return _bi.max(x, *a) if a else _bi.max(_flat(x))   # noqa: A001
def min(x, *a): return _bi.min(x, *a) if a else _bi.min(_flat(x))   # noqa: A001
def sum(x, *a, **k): return _bi.sum(_flat(x))                       # noqa: A001
def mean(x): return sum(x) / _bi.max(1, len(list(_flat(x))))
def amax(x): return max(x)
def amin(x): return min(x)


def argmax(x):
    v = list(_flat(x))
    return v.index(_bi.max(v))


def argmin(x):
    v = list(_flat(x))
    return v.index(_bi.min(v))
def arctan2(y, x): return _apply2(y, x, _math.atan2)
def isnan(x): return _apply1(x, lambda v: v != v)
def sinc(x): return _apply1(x, lambda v: 1.0 if v == 0 else _math.sin(pi * v) / (pi * v))


float64 = float
float32 = float
int32 = int
int64 = int
bool_ = bool
complex128 = complex


class random:
    @staticmethod
    def rand(*shape):
        n = shape[0] if shape else None
        return ndarray([_random.random() for _ in range(n)]) if n else _random.random()

    @staticmethod
    def randn(*shape):
        n = shape[0] if shape else None
        return (ndarray([_random.gauss(0, 1) for _ in range(n)]) if n
                else _random.gauss(0, 1))

    @staticmethod
    def randint(lo, hi=None, size=None):
        if hi is None:
            lo, hi = 0, lo
        if size is None:
            return _random.randrange(lo, hi)
        return ndarray([_random.randrange(lo, hi) for _ in range(size)])

    @staticmethod
    def seed(s=None):
        _random.seed(s)


def __getattr__(name):
    raise AttributeError(
        f"numpy.{name} is not available: qua-scope ships a minimal numpy "
        f"stand-in because numpy is not installed in this environment"
    )
