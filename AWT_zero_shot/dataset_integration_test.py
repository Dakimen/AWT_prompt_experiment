import torch
from collections import Counter

from data.datautils import build_dataset
from data.cls_to_names import get_classnames, CUSTOM_TEMPLATES

root = "/root/autodl-tmp/datasets"

classes = get_classnames("household_objects")

print("Registered classes:", classes)
print("Text template:", CUSTOM_TEMPLATES["household_objects"])

dataset = build_dataset(
    "household_objects",
    transform=None,
    data_root=root,
    mode="test"
)

print("Test images:", len(dataset))

labels = [int(label) for label in dataset.label_list]

print("Class distribution:", dict(Counter(labels)))

assert classes == ["chair", "lamp", "table", "vase"]
assert len(dataset) == 100
assert Counter(labels) == {0: 25, 1: 25, 2: 25, 3: 25}

# Verify image reading
image, label = dataset[0]

print("First image dimensions:", image.size)
print("First image label:", label.item())

assert image.mode == "RGB"

print("\nCustom dataset registered successfully!")