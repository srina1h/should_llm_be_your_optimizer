"""LLM-free tests for OPRO/SNAP2 seed construction (build_opro_seed)."""
import os
import random
import sys
import tempfile

# Put the package root on sys.path so `src.*`, `methods.*`, `prompts.*` resolve.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.utils.ezr import DATA, csv, d2h
from methods.opro import build_opro_seed


class Args:
    def __init__(self, **kw):
        self.label = 4
        self.last = 20
        self.opro_seed = "random"
        self.opro_seed_size = None
        for k, v in kw.items():
            setattr(self, k, v)


def _make_csv():
    # 2 features (x), 1 minimize objective (y-). 40 rows, deterministic.
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w") as f:
        f.write("X1,X2,Y-\n")
        for i in range(40):
            f.write(f"{i},{(i * 7) % 40},{(i - 20) ** 2}\n")
    return path


def _load():
    return DATA(csv(_make_csv()))


def test_random_partition():
    random.seed(123)
    d = _load()
    done, todo, traj = build_opro_seed(d, Args(opro_seed="random"))
    assert len(done) == 4, len(done)
    assert len(done) + len(todo) == len(d.rows)
    ids = {id(r) for r in done}
    assert all(id(r) not in ids for r in todo), "done/todo overlap"
    assert traj == [(r, d2h(d, r)) for r in done]
    print("test_random_partition OK")


def test_random_is_deterministic_under_seed():
    random.seed(7)
    d = _load()
    done1, _, _ = build_opro_seed(d, Args(opro_seed="random"))
    random.seed(7)
    d2 = _load()
    done2, _, _ = build_opro_seed(d2, Args(opro_seed="random"))
    assert [r[:] for r in done1] == [r[:] for r in done2]
    print("test_random_is_deterministic_under_seed OK")


def test_exploit_size_and_disjoint():
    random.seed(123)
    d = _load()
    done, todo, traj = build_opro_seed(d, Args(opro_seed="exploit"))
    assert len(done) == round(20 / 2), len(done)  # S = round(last/2) = 10
    assert len(done) + len(todo) == len(d.rows)
    ids = {id(r) for r in done}
    assert all(id(r) not in ids for r in todo), "done/todo overlap"
    assert traj == [(r, d2h(d, r)) for r in done]
    print("test_exploit_size_and_disjoint OK")


def test_exploit_best_d2h_monotone_vs_initial_label():
    # The exploit pre-pass only ADDS rows, so the best d2h over the S-row seed
    # can only be <= the best d2h over the first `label` rows it started from.
    random.seed(99)
    d = _load()
    done, _, _ = build_opro_seed(d, Args(opro_seed="exploit"))
    best_seed = min(d2h(d, r) for r in done)
    # Reconstruct the initial label rows under the same RNG sequence start.
    random.seed(99)
    d2 = _load()
    pool = list(d2.rows)
    random.shuffle(pool)
    initial_label = pool[:4]
    best_initial = min(d2h(d2, r) for r in initial_label)
    assert best_seed <= best_initial + 1e-9, (best_seed, best_initial)
    print("test_exploit_best_d2h_monotone_vs_initial_label OK")


def test_seed_size_override_clamped():
    random.seed(1)
    d = _load()
    done, _, _ = build_opro_seed(d, Args(opro_seed="exploit", opro_seed_size=1000))
    assert len(done) == 20, len(done)  # clamped to last
    print("test_seed_size_override_clamped OK")


if __name__ == "__main__":
    test_random_partition()
    test_random_is_deterministic_under_seed()
    test_exploit_size_and_disjoint()
    test_exploit_best_d2h_monotone_vs_initial_label()
    test_seed_size_override_clamped()
    print("ALL OK")
