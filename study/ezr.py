"""Distance, likelihood and projection routines from EZR (Tim Menzies, BSD-2)."""

import ast
import math
import random
from types import SimpleNamespace

the = SimpleNamespace(p=2, Half=128, Far=0.8, N=0.5, k=1, m=2)


def coerce(s):
    """Coerces strings to atoms."""
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def SYM(txt, at):
    """SYM columns incrementally summarizes a stream of symbols."""
    return SimpleNamespace(this=SYM, txt=txt, at=at, n=0, has={})


def NUM(txt, at):
    """NUM columns incrementally summarizes a stream of numbers."""
    return SimpleNamespace(
        this=NUM, txt=txt, at=at, n=0, hi=-1e30, lo=1e30, mu=0, m2=0, sd=0, maximize=txt[-1] != "-"
    )


def COLS(names):
    """Create columns (one for each string in `names`)."""
    cols = SimpleNamespace(names=names, x=[], y=[], all=[])
    for index, name in enumerate(names):
        col = (NUM if name[0].isupper() else SYM)(name, index)
        cols.all.append(col)
        if not name.endswith("X"):
            (cols.y if name.endswith(("!", "+", "-")) else cols.x).append(col)
    return cols


def DATA(src, rank=False):
    rows = iter(src)
    data = SimpleNamespace(cols=COLS(next(rows)), rows=[])
    for row in rows:
        data.rows.append([update(col, x) for col, x in zip(data.cols.all, row)])
    if rank:
        data.rows.sort(key=lambda row: d2h(data, row))
    return data


def update(col, x):
    if x == "?":
        return x
    col.n += 1
    if col.this is NUM:
        col.lo = min(x, col.lo)
        col.hi = max(x, col.hi)
        delta = x - col.mu
        col.mu += delta / col.n
        col.m2 += delta * (x - col.mu)
        col.sd = 0 if col.n < 2 else (col.m2 / (col.n - 1)) ** 0.5
    else:
        col.has[x] = col.has.get(x, 0) + 1
    return x


def clone(i, inits=(), rank=False):
    """Copy a DATA (same column structure, with different rows). Optionally, sort it."""
    return DATA([i.cols.names, *inits], rank=rank)


def norm(i, x):
    """Normalize `x` to 0..1"""
    return x if x == "?" else (x - i.lo) / (i.hi - i.lo - 1e-30)


def d2h(i, r):
    """distance to `heaven` (which is the distance of the `y` vals to the best values)."""
    n = sum((abs(norm(num, r[num.at]) - num.maximize) ** the.p for num in i.cols.y))
    return (n / len(i.cols.y)) ** (1 / the.p)


def dists(i, r1, r2):
    """Distances between two rows."""
    n = sum((dist(c, r1[c.at], r2[c.at]) ** the.p for c in i.cols.x))
    return (n / len(i.cols.x)) ** (1 / the.p)


def dist(i, x, y):
    """Distance between two values. Used by `dists()`."""
    if x == y == "?":
        return 1
    if i.this is SYM:
        return x != y
    x, y = (norm(i, x), norm(i, y))
    x = x if x != "?" else 1 if y < 0.5 else 0
    y = y if y != "?" else 1 if x < 0.5 else 0
    return abs(x - y)


def neighbors(i, r1, region=None):
    """Sort the `region` (default=`i.rows`),ascending, by distance to `r1`."""
    return sorted(region or i.rows, key=lambda r2: dists(i, r1, r2))


def branch(i, region=None, stop=None, rest=None, evals=1, before=None):
    """Recursively bi-cluster `region`, recurse only down the best half."""
    region = region or i.rows
    if not stop:
        random.shuffle(region)
    stop = stop or 2 * len(region) ** the.N
    rest = rest or []
    if len(region) > stop:
        lefts, rights, left, _ = half(i, region, True, before)
        return branch(i, lefts, stop, rest + rights, evals + 1, left)
    else:
        return (region, rest, evals)


def half(i, region, sortp=False, before=None):
    """Split the `region` in half according to each row's distance to two distant points. Used by `branch()`."""
    mid = int(len(region) // 2)
    left, right, C = _twoFaraway(i, region, sortp=sortp, before=before)

    def project(row1):
        return (dists(i, row1, left) ** 2 + C**2 - dists(i, row1, right) ** 2) / (2 * C + 1e-30)

    tmp = sorted(region, key=project)
    return (tmp[:mid], tmp[mid:], left, right)


def _twoFaraway(i, region, before=None, sortp=False):
    """Find two distant points within the `region`. Used by `half()`."""
    region = random.choices(region, k=min(the.Half, len(region)))
    x = before or _faraway(i, random.choice(region), region)
    y = _faraway(i, x, region)
    if sortp and d2h(i, y) < d2h(i, x):
        x, y = (y, x)
    return (x, y, dists(i, x, y))


def _faraway(i, r1, region):
    """Find something far away from `r1` with the `region`. Used by `_twoFaraway()`."""
    farEnough = int(len(region) * the.Far)
    return neighbors(i, r1, region)[farEnough]


def loglikes(i, r, nall, nh):
    """Likelihood of a `row` belonging to a DATA."""
    prior = (len(i.rows) + the.k) / (nall + the.k * nh)
    likes = [like(c, r[c.at], prior) for c in i.cols.x if r[c.at] != "?"]
    return sum((math.log(x) for x in likes + [prior] if x > 0))


def like(i, x, prior):
    """Likelihood of `x` belonging to a col. Used by `loglikes()`."""
    return _like4num(i, x) if i.this is NUM else (i.has.get(x, 0) + the.m * prior) / (i.n + the.m)


def _like4num(i, x):
    """Likelihood of `x` belonging to a NUM. Used by `like()`."""
    v = i.sd**2 + 1e-30
    nom = math.e ** (-1 * (x - i.mu) ** 2 / (2 * v)) + 1e-30
    denom = (2 * math.pi * v) ** 0.5
    return min(1, nom / (denom + 1e-30))
