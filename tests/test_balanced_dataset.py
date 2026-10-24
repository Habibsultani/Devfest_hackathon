from collections import Counter

import pytest

from scripts.prepare_balanced_dataset import choose_balanced_rows


def rows():
    result = [{"source_path": f"normal-{i}", "group_id": f"normal-{i}", "label": "0", "original_split": "train"}
              for i in range(12)]
    result.extend({"source_path": f"defect-{group}-{i}", "group_id": f"defect-{group}", "label": "1", "original_split": "test"}
                  for group in range(4) for i in range(3))
    return result


def test_balance_quotas_and_group_separation_are_enforced():
    selected, audit = choose_balanced_rows(rows(), train_normal=6, validation_per_label=3, test_per_label=3)
    assert Counter((r["split"], int(r["label"])) for r in selected) == {
        ("train", 0): 6, ("validation", 0): 3, ("validation", 1): 3, ("test", 0): 3, ("test", 1): 3}
    owners = {}
    for row in selected:
        assert owners.setdefault(row["group_id"], row["split"]) == row["split"]
    assert len({r["source_path"] for r in selected}) == len(selected)
    assert choose_balanced_rows(rows(), train_normal=6, validation_per_label=3, test_per_label=3) == (selected, audit)


def test_balance_fails_instead_of_oversampling_or_leaking():
    with pytest.raises(ValueError, match="Not enough normal"):
        choose_balanced_rows(rows(), train_normal=13)
    insufficient = [r for r in rows() if r["label"] == "0" or r["group_id"] == "defect-0"]
    with pytest.raises(ValueError, match="two defect source groups"):
        choose_balanced_rows(insufficient, train_normal=6, validation_per_label=1, test_per_label=1)
