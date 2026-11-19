
import csv
import json
from pathlib import Path
from collections import Counter

root = Path("/root/autodl-tmp/datasets/dataset")
manifest = root / "metadata" / "dataset_manifest.csv"

class_to_id = {
    "chair": 0,
    "lamp": 1,
    "table": 2,
    "vase": 3
}

split = {"train": [], "val": [], "test": []}
counts = Counter()

with manifest.open(encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        label = row["class"].strip()
        group = row["split"].strip()
        path = row["relative_path"].strip()

        assert label in class_to_id, label
        assert group in ("development", "test"), group
        assert (root / path).is_file(), path

        target_split = "val" if group == "development" else "test"

        split[target_split].append([
            path,
            class_to_id[label],
            label
        ])
        counts[(target_split, label)] += 1

assert len(split["test"]) == 100
assert len(split["val"]) == 12

for label in class_to_id:
    assert counts[("test", label)] == 25
    assert counts[("val", label)] == 3

output = root / "metadata" / "split_household_objects.json"
output.write_text(
    json.dumps(split, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("Saved:", output)
print("Test:", len(split["test"]))
print("Development:", len(split["val"]))
print("Class counts:", dict(counts))
