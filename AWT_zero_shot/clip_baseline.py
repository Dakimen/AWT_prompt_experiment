
import argparse
import csv
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from clip import clip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    manifest = root / "metadata" / "dataset_manifest.csv"

    classes = ["chair", "lamp", "table", "vase"]
    class_to_id = {name: i for i, name in enumerate(classes)}

    with manifest.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    test_rows = [r for r in rows if r["split"].strip() == "test"]
    print("Test images:", len(test_rows))

    assert len(test_rows) == 100, "Expected exactly 100 test images"
    assert set(r["class"] for r in test_rows) == set(classes)
    assert Counter(r["class"] for r in test_rows) == {
        name: 25 for name in classes
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)
    model.eval()

    prompts = [f"a photo of a {name}." for name in classes]
    tokens = clip.tokenize(prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

    predictions = []
    correct = 0
    per_class = Counter()

    with torch.no_grad():
        for row in test_rows:
            image_path = root / row["relative_path"]
            assert image_path.is_file(), f"Missing: {image_path}"

            with Image.open(image_path) as image:
                image_tensor = preprocess(
                    image.convert("RGB")
                ).unsqueeze(0).to(device)

            image_features = model.encode_image(image_tensor).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarities = (image_features @ text_features.T)[0]
            prediction_id = int(similarities.argmax().item())
            predicted = classes[prediction_id]
            actual = row["class"]

            is_correct = predicted == actual
            correct += int(is_correct)
            per_class[actual] += int(is_correct)

            
            # Signed Ground-Truth Similarity Margin (SGTM)
            true_id = class_to_id[actual]
            true_score = similarities[true_id]

            competitor_scores = similarities.clone()
            competitor_scores[true_id] = float("-inf")
            strongest_competitor = competitor_scores.max()

            signed_margin = float(
                (true_score - strongest_competitor).item()
            )

            predictions.append({
                "image_id": row["image_id"],
                "relative_path": row["relative_path"],
                "actual": actual,
                "predicted": predicted,
                "correct": int(is_correct),
                "signed_margin": signed_margin
            })


    output = Path("baseline_predictions.csv")
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)

    print(f"\nOverall accuracy: {correct}/{len(test_rows)} "
          f"= {100 * correct / len(test_rows):.2f}%")

    print("\nPer-class accuracy:")
    for name in classes:
        print(f"  {name}: {per_class[name]}/25 "
              f"= {per_class[name] * 4:.2f}%")
    # Summarize signed margins
    margins = [row["signed_margin"] for row in predictions]
    mean_margin = sum(margins) / len(margins)

    print(f"\nMean SGTM: {mean_margin:.6f}")
    print(f"Minimum SGTM: {min(margins):.6f}")
    print(f"Maximum SGTM: {max(margins):.6f}")
    print(f"Negative margins: {sum(m < 0 for m in margins)}")

    print(f"\nPredictions saved to {output}")


if __name__ == "__main__":
    main()
