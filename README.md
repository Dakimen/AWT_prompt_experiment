# LLM Prompting Strategies for Zero-Shot Image Classification

## Overview

This repository contains a course experiment investigating how different LLM-generated class description strategies affect zero-shot image classification using CLIP and AWT.

The experiment uses GPT-4o to generate class descriptions under four prompting conditions and evaluates their effect on classification accuracy, signed ground-truth classification margins, and softmax-based cross-entropy.

**This project is an adaptation of the original AWT implementation, not an independent implementation of the AWT algorithm.**

## Original Work and Attribution

This project builds upon [AWT: Transferring Vision-Language Models via Augmentation, Weighting, and Transportation](https://github.com/MCG-NJU/AWT), developed by Yuhan Zhu, Yuyang Ji, Zhiyu Zhao, Gangshan Wu, and Limin Wang (NeurIPS 2024).

The original AWT architecture, optimal transport matching, and two-stage description-generation framework are credited to their respective authors.

The modifications in this repository were made for an MSc course experiment and are not part of the official AWT release.

## Experimental Dataset

A custom household-object dataset was collected from Wikimedia Commons.

- Four categories: chair, lamp, table, vase
- 100 test photographs, 25 per class
- 12 additional development photographs
- Image provenance, attribution, and licensing information recorded in the dataset manifest

## Prompting Conditions

| Condition              | Description                                                           |
| ---------------------- | --------------------------------------------------------------------- |
| A — Dataset-aware      | Original AWT-style prompting with correct household dataset context   |
| B — No context         | Question generation without dataset context                           |
| C — Mismatched context | Question generation using unrelated satellite-imagery dataset context |
| D — Non-visual         | Generation emphasizing function, purpose, and semantic associations   |

All conditions use GPT-4o (`gpt-4o`) with 50 class descriptions per category during evaluation.

## Implementation Changes

The original AWT repository was adapted to:

- Register the custom household-object dataset.
- Support local LLM description generation through the OpenAI API.
- Preserve and separately evaluate four prompting conditions.
- Record per-image class scores and predictions.
- Calculate signed classification margins and softmax-based cross-entropy.
- Export aggregate metrics and per-image CSV results.

## Results

| Condition | Accuracy | Mean signed margin | Cross-entropy |
| --------- | -------: | -----------------: | ------------: |
| A         |      99% |            0.05279 |       0.92768 |
| B         |      98% |            0.05325 |       0.92183 |
| C         |      96% |            0.05387 |       0.92023 |
| D         |      98% |            0.05302 |       0.92764 |

Cross-entropy is calculated using a fixed softmax temperature of 0.1 applied to normalized AWT scores.

The results are exploratory and limited to a small custom dataset. They should not be interpreted as evidence of general superiority across datasets.

## Reproducing the Experiment

The repository includes the modified AWT implementation, dataset metadata, generated LLM descriptions for all four experimental conditions, and evaluation scripts. **No OpenAI API access is required to reproduce the classification results.**

1. **Reconstruct the dataset**
   - Download the photographs from the Wikimedia Commons sources listed in `dataset_manifest.csv`. Images are not redistributed in this repository.
   - Preserve the filenames, relative paths, and exact image resolutions recorded in the manifest.
   - Reconstruct the original dataset directory structure under `dataset/household_objects/`, with 100 test images (25 per class) and 12 development images.
   - Keep `metadata/split_household_objects.json`, which defines the dataset splits, class labels, and test sample order.

2. **Dataset integration**
   - The custom dataset is already registered in `AWT_zero_shot/data/datautils.py` and `AWT_zero_shot/data/cls_to_names.py`.
   - Classes are ordered as `chair`, `lamp`, `table`, and `vase`, using the CLIP prompt template `a photo of a {}`.
   - Ensure the reconstructed dataset paths match those referenced by the dataset loader.

3. **Run the standard CLIP baseline**

   From `AWT_zero_shot/`, execute:

   ```bash
   python clip_baseline.py --dataset-root /path/to/dataset
   ```

   The baseline outputs individual predictions to `baseline_predictions.csv`. The original experiment achieved 99% top-1 accuracy.

4. **Extract image features**

   From `AWT_zero_shot/`, execute:

   ```bash
   python pre_extract.py /path/to/dataset-parent \
       --test_set household_objects \
       --arch ViT-B/16 \
       --batch-size 50 \
       --workers 4 \
       --seed 0
   ```

   The dataset parent directory must contain the `dataset/` folder. Keep `shuffle=False` during test feature extraction to preserve the correspondence between feature indices and image filenames.

5. **Evaluate the four prompting conditions**

   The pre-generated description files are included in `AWT_zero_shot/descriptions/image_datasets/`:
   - `A_dataset_aware_household_objects.json` — Correct dataset context
   - `B_no_context_household_objects.json` — No dataset context
   - `C_mismatched_context_household_objects.json` — Incorrect dataset context
   - `D_non_visual_household_objects.json` — Non-visual semantic descriptions

   Before evaluating each condition, copy its corresponding JSON file to a clean `household_objects.json` in the same directory.

   For example, to evaluate condition A:

   ```bash
   cp descriptions/image_datasets/A_dataset_aware_household_objects.json \
      descriptions/image_datasets/household_objects.json

   python evaluate.py \
       --test_set household_objects \
       --arch ViT-B/16 \
       --descriptor_path ./descriptions/image_datasets \
       --num_descriptor 50 \
       --seed 0 \
       --condition A
   ```

   Repeat for B, C, and D, changing both the description file and `--condition` argument.

6. **Examine the results**
   - Results are saved to `experiment_results/<condition>/metrics.json` and `predictions.csv`.
   - Reported metrics include top-1 accuracy, per-class accuracy, mean signed ground-truth classification margin, and cross-entropy using a fixed softmax temperature of 0.1.
   - The prediction CSVs contain per-image class scores, allowing analysis of misclassified images and differences between prompting conditions.
   - Existing result files must be moved or removed before rerunning a condition, because the modified evaluator prevents accidental overwriting.

**Notes:** The experiment uses CLIP ViT-B/16 and the first 50 descriptions per class. All four conditions must use the same extracted image features and evaluation settings. The original GPT-4o descriptions are provided, so no additional description-generation step is necessary.

Generated description JSON files and evaluation results are stored separately for each experimental condition.

## License

This repository contains modified code from AWT, originally distributed under the Apache License 2.0. See the retained `LICENSE` file and upstream repository for applicable terms.

## References

Zhu, Y., Ji, Y., Zhao, Z., Wu, G., and Wang, L. (2024). _AWT: Transferring Vision-Language Models via Augmentation, Weighting, and Transportation_. NeurIPS 2024.

Original implementation: https://github.com/MCG-NJU/AWT
