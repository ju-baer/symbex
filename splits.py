"""
SYMBEX v2 — Honest Dataset Splits
===================================

FIX for the v1 flaw: v1 split by *transform type* (base->train, SP->val,
SB->test) while keeping the same (template, seed) pair across all three
splits. That means every test item was a one-edge-edit away from an item
the "model" supposedly never saw in train — 100% of test (template, seed)
keys appeared in train. This is not held-out evaluation.

v2 splits along TWO independent axes simultaneously, producing a split that
is actually defensible:

  Axis 1 — TEMPLATE HOLDOUT: 2 of the 10 templates (one chosen to stress
  safety, one to stress mechanistic probing) are reserved ENTIRELY for test.
  Their base/SP/SB instances never appear in train or validation under any
  seed. This is what makes "test" a real generalization check on novel
  task structure, not just a novel edge-edit of a familiar graph.

  Axis 2 — SEED HOLDOUT on the remaining 8 templates: seeds are partitioned
  into disjoint train/val/test seed ranges (e.g. seeds 0-5 train, 6-7 val,
  8-9 test), so even within a shared template, no two splits see the exact
  same sampled instance (same capacity values, same names, same costs).

Within EVERY split, the base/SP/SB grouping is preserved as a *property*
(so invariance/asymmetry analysis can still run inside any split — e.g. you
can still measure "does this agent's success rate on SP variants match its
success rate on base instances" using only test data), but it is no longer
the axis that defines the split boundary.
"""

from typing import List, Dict
from collections import defaultdict
from core_schema import BenchmarkItem


# Templates fully held out for test (never seen, at any seed, in train/val).
# Chosen to stress two distinct capabilities: D2 (RBAC / safety reasoning)
# and E1 (mechanistic edge-sensitivity) so the held-out generalization test
# is not confined to one behavioral family.
HELDOUT_TEMPLATES = {"D2", "E1"}

# Of the remaining templates, seed ranges are partitioned disjointly.
# With NUM_SEEDS_HF seeds total per template, this carves:
#   [0, train_frac)                -> train
#   [train_frac, train_frac+val_n) -> validation
#   [train_frac+val_n, num_seeds)  -> test
def _seed_partition(num_seeds: int):
    train_n = max(1, round(num_seeds * 0.6))
    val_n = max(1, round(num_seeds * 0.2))
    test_n = max(1, num_seeds - train_n - val_n)
    train_seeds = set(range(0, train_n))
    val_seeds = set(range(train_n, train_n + val_n))
    test_seeds = set(range(train_n + val_n, train_n + val_n + test_n))
    return train_seeds, val_seeds, test_seeds


def split_dataset(items: List[BenchmarkItem], num_seeds: int) -> Dict[str, List[BenchmarkItem]]:
    """Returns {'train': [...], 'validation': [...], 'test': [...]} with NO
    (template, seed) overlap between any two splits, and held-out templates
    appearing ONLY in test.
    """
    train_seeds, val_seeds, test_seeds = _seed_partition(num_seeds)

    splits = {"train": [], "validation": [], "test": []}
    for item in items:
        if item.template in HELDOUT_TEMPLATES:
            splits["test"].append(item)
        elif item.seed in train_seeds:
            splits["train"].append(item)
        elif item.seed in val_seeds:
            splits["validation"].append(item)
        elif item.seed in test_seeds:
            splits["test"].append(item)
        else:
            # Defensive: any seed not covered by the partition (shouldn't
            # happen given _seed_partition covers [0, num_seeds)) goes to test.
            splits["test"].append(item)
    return splits


def verify_no_leakage(splits: Dict[str, List[BenchmarkItem]]) -> Dict:
    """Returns a leakage report. A clean split has zero overlap and held-out
    templates appearing only in test."""
    keys = {
        name: set((it.template, it.seed) for it in items)
        for name, items in splits.items()
    }
    train_test_overlap = keys["train"] & keys["test"]
    train_val_overlap = keys["train"] & keys["validation"]
    val_test_overlap = keys["validation"] & keys["test"]

    heldout_outside_test = set()
    for name in ("train", "validation"):
        for it in splits[name]:
            if it.template in HELDOUT_TEMPLATES:
                heldout_outside_test.add((name, it.template, it.seed))

    return {
        "train_test_key_overlap": len(train_test_overlap),
        "train_val_key_overlap": len(train_val_overlap),
        "val_test_key_overlap": len(val_test_overlap),
        "heldout_template_leak_into_train_or_val": len(heldout_outside_test),
        "heldout_leak_examples": list(heldout_outside_test)[:5],
        "is_clean": (
            len(train_test_overlap) == 0
            and len(train_val_overlap) == 0
            and len(val_test_overlap) == 0
            and len(heldout_outside_test) == 0
        ),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/claude/symbex_v2")
    from generators_v2 import generate_full_dataset

    NUM_SEEDS = 10
    items = generate_full_dataset(NUM_SEEDS)
    splits = split_dataset(items, NUM_SEEDS)

    print(f"Split sizes: train={len(splits['train'])}  val={len(splits['validation'])}  test={len(splits['test'])}")

    report = verify_no_leakage(splits)
    print("\nLeakage report:")
    for k, v in report.items():
        print(f"  {k}: {v}")

    print(f"\nClean split: {report['is_clean']}")

    # Show which templates appear in each split
    from collections import Counter
    for name, sub in splits.items():
        tmpl_counts = Counter(it.template for it in sub)
        print(f"\n{name} templates: {dict(sorted(tmpl_counts.items()))}")
