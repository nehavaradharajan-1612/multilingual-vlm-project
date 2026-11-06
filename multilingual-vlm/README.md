# Multilingual Vision-Language Classification

**Classify handwritten Chinese numerals using a trainable PyTorch image encoder and frozen multilingual text embeddings, then measure how image corruption affects performance.**

This project provides a complete command-line workflow for dataset acquisition, writer-disjoint partitioning, multilingual prompt encoding, model training, checkpoint selection, clean test evaluation, corruption testing, and single-image inference.

> **Scope:** Chinese handwriting is the visual input. Chinese, English, and Spanish are the languages of the class descriptions. The dataset does not contain handwriting in three different scripts. The supplied résumé's **96% validation accuracy is not assumed, hard-coded, or guaranteed**. This is a newly implemented project, not a recovery of an unknown earlier implementation. Read `VALIDATION.md` for results actually established during delivery.

## Measured delivery experiment

A real five-epoch noise-augmented run completed on the full training split. It reached **86.89% mean clean validation accuracy** and **88.64% mean clean test accuracy**, averaged across Chinese, English, and Spanish prompt results. The trained checkpoint, cached real text embeddings, complete test predictions, and 48-condition robustness results are included. See `VALIDATION.md` for per-language scores and limitations. These results do not establish the supplied 96% claim.

## 1. Dataset and task

The project uses **[Chinese MNIST](https://www.kaggle.com/datasets/gpreda/chinese-mnist)**. The downloaded release contains 15,000 images and metadata for 100 writers, with 10 examples of each of 15 numeral characters per writer. The images are 64 × 64 pixels.

| Class ID | Character | Numeric value | English name |
|---:|:---:|---:|---|
| 0 | 零 | 0 | zero |
| 1 | 一 | 1 | one |
| 2 | 二 | 2 | two |
| 3 | 三 | 3 | three |
| 4 | 四 | 4 | four |
| 5 | 五 | 5 | five |
| 6 | 六 | 6 | six |
| 7 | 七 | 7 | seven |
| 8 | 八 | 8 | eight |
| 9 | 九 | 9 | nine |
| 10 | 十 | 10 | ten |
| 11 | 百 | 100 | one hundred |
| 12 | 千 | 1,000 | one thousand |
| 13 | 万 | 10,000 | ten thousand |
| 14 | 亿 | 100,000,000 | one hundred million |

The importer checks the actual CSV's `code`, `character`, and `value` against this mapping. It resolves filenames from `suite_id`, `sample_id`, and `code`, decodes every image, and rejects duplicate decoded RGB images and duplicate identifiers. It does not silently drop unreadable examples or label conflicts.

### Why writer-disjoint partitions matter

Randomly splitting individual images allows the same writer's style into training and validation. Here `suite_id` is used as the writer group. With seed 42 and the complete release, writers are assigned to:

- **Training:** 70 writers, 10,500 images.
- **Validation:** 15 writers, 2,250 images.
- **Test:** 15 writers, 2,250 images.

Every writer belongs to exactly one split. Every split must contain all 15 classes. The split is generated with a seeded shuffle of sorted writer identifiers, recorded in the manifest, and reused for every experiment. This is a project-defined split, not a claimed official benchmark partition.

## 2. Architecture: what is being trained?

The model has two parts:

1. A **frozen pretrained multilingual text encoder**: `sentence-transformers/clip-ViT-B-32-multilingual-v1`. It converts 15 class prompts in each of three languages into 45 text vectors. These are cached before image training.
2. A **trainable CNN image encoder** built in PyTorch. Three convolutional blocks learn visual stroke patterns; an MLP maps the image representation into the same dimensional space as the text vectors.

The pretrained text model is documented to align multilingual text to CLIP's embedding space. **This project does not use the pretrained CLIP image encoder.** It learns its own smaller image encoder from the labeled Chinese numeral images. It is therefore a supervised vision-language classifier with a frozen language branch—not end-to-end CLIP fine-tuning, a generative VLM, or large-scale contrastive pretraining.

For image embedding `z` and language-specific class vector `t[l,c]`, the score is:

```text
logit[l,c] = 20 × cosine_similarity(z, t[l,c])
```

Both image and text vectors are L2-normalized. The factor 20 is fixed. The training objective is the mean of three cross-entropy losses, one for each prompt language. This ensures the image representation is optimized against all three sets of text descriptions.

The text branch remains frozen. Its vectors are buffers in the saved model, so inference does not need to download the language model again. There is no learned 15-way linear classification head: the class directions come from the text embeddings.

### Image encoder

| Stage | Operation | Output shape per image |
|---|---|---|
| Input | RGB image, resize to 64 × 64, normalize to [-1, 1] | 3 × 64 × 64 |
| Block 1 | Conv 3→32, GroupNorm, GELU, max pooling | 32 × 32 × 32 |
| Block 2 | Conv 32→64, GroupNorm, GELU, max pooling | 64 × 16 × 16 |
| Block 3 | Conv 64→128, GroupNorm, GELU, max pooling | 128 × 8 × 8 |
| Projection | Flatten, Linear→256, GELU, dropout 0.2, Linear→text dimension | Text embedding dimension |
| Scoring | Normalized similarities to 45 frozen prototypes | 3 × 15 logits |

Spatial information is retained through flattening because stroke location matters for character recognition. No flips or rotations are applied: those transformations can alter a character's meaning.

### Multilingual prompt design

Examples for the first class:

| Language | Prompt |
|---|---|
| Chinese | 手写汉字“零”。 |
| English | The handwritten Chinese numeral for zero. |
| Spanish | El numeral chino manuscrito que significa cero. |

All prompts and their order are saved in the prototype metadata. English and Spanish descriptions deliberately do not contain the Chinese glyph. The three prompt sets convey the same class meanings but are not identical strings. Language differences can reflect translation and representation quality; they do not measure performance on separate visual-language corpora.

## Included notebook

`Run_Project.ipynb` provides the same real-data workflow in notebook form. Open it with the extracted project root as the working directory. Its cells are unexecuted templates; measured delivery results are stored separately.

## 3. Installation

Use Python 3.11 or newer in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[data,test]'
```

Windows PowerShell activation: `.venv\Scripts\Activate.ps1`.

For CPU-only PyTorch, install it first using the appropriate command from [PyTorch](https://pytorch.org/get-started/locally/). For a CUDA machine, choose a PyTorch build compatible with your driver and use `--device cuda` in training/evaluation. CPU is the default. No torchvision dependency is required by the project itself.

If `vlm` is not found, use `python -m vlm.cli` instead. Run commands from the repository root. All output directories must be new; choose a new run name instead of overwriting an experiment.

## 4. Prepare the real dataset

```bash
vlm download
```

This calls KaggleHub for `gpreda/chinese-mnist` and prints the downloaded directory. Kaggle may require authentication or acceptance of dataset terms in your environment. Do not put credentials in this repository. You can also download and extract the archive manually from the dataset page.

Then pass the directory containing `chinese_mnist.csv` and the image tree:

```bash
vlm prepare --root /absolute/path/to/downloaded/chinese-mnist \
  --out data/prepared --seed 42
```

The search is recursive, so the release's nested `data/data/` image layout is supported. Outputs are `manifest.csv` and `data_card.json`. Paths in the manifest are absolute to avoid ambiguity; if you relocate the raw data, prepare a fresh manifest using the same seed. Evaluation checks a portable fingerprint of image IDs, labels, writer groups, splits, and decoded-image hashes, excluding filesystem paths. Re-run preparation after changing any source image.

The original images are not redistributed in this ZIP. Follow the publisher's terms when downloading or sharing them. Local checksums and split metadata identify the experiment; they do not substitute for dataset permission.

## 5. Encode text prompts once

**Shortcut:** this delivery includes the real cached prompt vectors in `artifacts/text/`. Use `--prototypes artifacts/text` in the training commands to reuse the exact delivery embeddings without a language-model download. The commands below show how to regenerate them.

```bash
vlm text --out data/text
```

The command downloads the pretrained multilingual sentence transformer, encodes the 45 prompts on CPU, and saves:

- `prototypes.npz`: the numerical language-by-class embedding array.
- `metadata.json`: model identifier, requested revision, languages, prompts, and whether this is synthetic demo data.

For a repeatable model version, pin a verified Hugging Face commit with `--revision COMMIT_HASH`. The project does not invent or silently claim a pinned commit when none is supplied. Preserve the generated prototype files; their SHA-256 is recorded in the training run. The delivery experiment used downloaded model revision `58edf8cada9e398793dca955574a48cbb7f18be2`, recorded from the local model cache. Only these small cached vectors are needed after encoding.

## 6. Train the model

Start with a clean-image baseline:

```bash
vlm train --manifest data/prepared/manifest.csv --prototypes data/text \
  --out runs/clean_seed42 --epochs 30 --batch-size 64 --lr 0.001 --seed 42
```

Then train a separate noise-augmented model using the same manifest and seed:

```bash
vlm train --manifest data/prepared/manifest.csv --prototypes data/text \
  --out runs/augmented_seed42 --epochs 30 --batch-size 64 --lr 0.001 \
  --seed 42 --augment
```

Defaults: AdamW, learning rate 0.001, weight decay 0.0001, gradient-norm clipping at 1, and early-stopping patience of seven epochs. No mixed precision or scheduler is used. The image encoder is initialized from scratch. The best checkpoint maximizes **mean clean validation accuracy across Chinese, English, and Spanish prompts**. Ties retain the earlier checkpoint. The mean counts each validation image once per language; it is not the accuracy of an ensemble prediction.

Training uses the training partition only. Validation is clean, fixed, and used for checkpoint selection. The test partition is loaded only by the separate evaluation command. Do not repeatedly inspect test outcomes to choose training settings.

The code seeds Python, NumPy, PyTorch, the training sampler, and per-image corruption choices. It enables deterministic PyTorch algorithms, sets two CPU compute threads, and uses zero data-loader workers. A CUDA backend may require `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python starts, or may reject an operation lacking a deterministic implementation. Record hardware, package versions, and commands; reproducibility across different devices is not guaranteed to be bit-identical.

## 7. Clean and corrupted test evaluation

```bash
vlm evaluate --manifest data/prepared/manifest.csv \
  --checkpoint runs/clean_seed42/best.pt --out runs/clean_test --robust

vlm evaluate --manifest data/prepared/manifest.csv \
  --checkpoint runs/augmented_seed42/best.pt --out runs/augmented_test --robust
```

For each checkpoint, the evaluation includes one clean condition and **15 corrupted conditions**: five corruption types at three severities. Each condition is evaluated with all three prompt languages, producing **48 metric rows**. The same corrupted image is used for all languages and both models when their seed is the same.

| Corruption | Severity 1 | Severity 2 | Severity 3 |
|---|---:|---:|---:|
| Gaussian pixel noise | SD 10 | SD 20 | SD 35 |
| Salt-and-pepper | 2% total pixels | 5% | 10% |
| Gaussian blur | Radius 0.5 | Radius 1.0 | Radius 1.5 |
| JPEG recompression | Quality 70 | Quality 40 | Quality 15 |
| Black square occlusion | Side 15% of image | Side 25% | Side 35% |

Gaussian SD is on the 0–255 pixel scale. Salt and pepper each receive half the stated rate. Occlusion percentages are **side lengths**, not image-area percentages. Perturbations are applied at native resolution, one at a time, before resizing and normalization. They are controlled synthetic corruptions, not a claim that the source dataset was collected under those noise conditions.

With `--augment`, each training image receives a deterministic epoch-dependent choice among clean and the five corruptions; non-clean examples use severity 1 or 2. Severity 3 is never used for training augmentation. Validation receives no augmentation.

Each corruption has one deterministic realization per image per severity. The seed is a SHA-256-derived function of run seed, sample ID, corruption, and severity, so ordering and batching do not change it. This is not a multi-seed robustness confidence interval or an adversarial attack benchmark.

Metrics include accuracy, balanced accuracy, macro F1, and accuracy drop in percentage points relative to that language's clean condition. A negative drop is possible if a perturbation happens to improve classification. Do not clamp or hide it. Softmax scores are uncalibrated closed-set scores; this system has no unknown-class rejection mechanism.

## 8. Predict a new image

```bash
vlm predict --checkpoint runs/augmented_seed42/best.pt \
  --image /absolute/path/to/character.jpg
```

The result contains the predicted class ID, Chinese character, and all class probabilities for each prompt language. The same 15-class inventory is always used. Arbitrary Chinese words, multi-character strings, scene text, and non-Chinese writing are outside the trained task.

The delivery checkpoint is included as `artifacts/best.pt`; use its measured limitations in `VALIDATION.md`. The checkpoint includes frozen text prototypes and image-model weights. Loading uses PyTorch's `weights_only=True`; load only artifacts from sources you trust.

## 9. Outputs and report figures

| File | Contents |
|---|---|
| `manifest.csv` | Image IDs, paths, targets, writers, split membership, decoded-image hashes |
| `data_card.json` | Dataset identity, counts, class order and prompts |
| `history.csv` | Training loss, validation accuracy per language, mean validation accuracy |
| `best.pt` | Selected vision-model weights, text vectors, provenance and selection score |
| `run.json` | Training settings, versions, seed, sample counts and prototype checksum |
| `manifest_snapshot.csv` | Manifest used by the training run |
| `metrics.csv` | Per-language, per-corruption test metrics and clean-relative accuracy drop |
| `predictions.csv` | Individual predictions, actual class and corruption identifiers |
| `confusion_zh.csv`, `confusion_en.csv`, `confusion_es.csv` | Clean test confusion matrices in class-ID order |
| `evaluation.json` | Checkpoint checksum, test size, corruption seed and evidence type |
| `learning_curve.png` | Training loss and clean validation accuracy per language |
| `confusion_matrices.png` | Three clean test confusion matrices |
| `robustness.png` | Three panels comparing accuracy across corruption severity by language |

For a report, use `robustness.png` for noise analysis and `learning_curve.png` for convergence. Confusion CSV rows are true classes and columns are predicted classes, both 0–14. Compare clean and augmented runs using identical split and corruption settings. Report clean accuracy as well as noisy performance; improvement under noise can come at a clean-performance cost.

## 10. Offline smoke test and automated checks

The offline demo does not download data or a language model:

```bash
vlm demo --out data/demo
vlm text --out data/demo_text --demo
vlm train --manifest data/demo/manifest.csv --prototypes data/demo_text \
  --out runs/demo --epochs 2 --batch-size 32
vlm evaluate --manifest data/demo/manifest.csv --checkpoint runs/demo/best.pt \
  --out runs/demo_test --robust
python -m pytest -q
```

It generates geometric images and identity-vector class prototypes. These are explicitly flagged as synthetic in saved metadata. The demo tests the execution path and **does not validate Chinese recognition or multilingual representations**. Never report its accuracy as the real project's result.

Tests cover corruption determinism, parameter validation, writer separation, leakage rejection, prompt alignment, image-encoder gradients with frozen text vectors, and checkpoint round-trip equality.

## 11. How to describe the completed work honestly

After running the real experiment, use measured values from `history.csv` and `metrics.csv`:

> Trained a PyTorch vision-language classifier for 15 handwritten Chinese numeral classes using frozen multilingual text prototypes and writer-disjoint evaluation, achieving **[measured]% mean validation accuracy** across Chinese, English, and Spanish prompts.
>
> Evaluated classification robustness under Gaussian noise, salt-and-pepper noise, blur, JPEG compression, and occlusion at three severity levels, reporting per-language accuracy, macro F1, and clean-relative degradation.

Only insert **96%** if the recorded validation results establish it, and specify whether the number is Chinese-prompt accuracy or the three-language mean. Do not substitute test accuracy for validation accuracy. Do not call the cached-text workflow end-to-end language-model fine-tuning. Keep Jan–Feb 2025 dates only if they accurately describe work you actually did then.

## 12. Limitations and next experiments

- The model classifies 15 isolated Chinese numeral characters; it is not full OCR or multilingual scene-text recognition.
- Language is supplied through fixed class prompts. The model does not consume a user-written description accompanying each image, generate captions, or reason conversationally.
- Numeral meanings can be close in the text embedding space. Cross-language prompt geometry can constrain recognition; the model may underperform an ordinary visual classifier.
- A useful next ablation is the same CNN with a conventional 15-way head. That baseline is not implemented or claimed in this package.
- The architecture and corruption severities are project choices, not a reproduced paper protocol. Real capture noise, distribution shift, and unseen writers outside this dataset require further evaluation.
- One train/validation/test partition and one corruption realization per condition do not establish statistical significance. Repeat with predeclared seeds before making broad robustness claims.
- Model download versions and dataset terms may change. Preserve prototypes, manifests, checkpoints and environment versions for the experiment you report.

## 13. Code map

| Module | Purpose |
|---|---|
| `vlm/data.py` | Class vocabulary, dataset validation, writer splits and synthetic generator |
| `vlm/text.py` | Multilingual prompt embeddings and their metadata |
| `vlm/model.py` | CNN encoder and text-prototype similarity classifier |
| `vlm/noise.py` | Reproducible corruption implementations |
| `vlm/engine.py` | Training, checkpointing, metrics, plotting and prediction |
| `vlm/cli.py` | All command-line entry points |
| `tests/test_project.py` | Automated correctness checks |
| `VALIDATION.md` | Checks and actual experiments completed for this delivery |

## 14. References

1. Chinese MNIST dataset: https://www.kaggle.com/datasets/gpreda/chinese-mnist
2. Multilingual CLIP text model card, alignment method and supported languages: https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1
3. Sentence Transformers pretrained image/text models: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
4. Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, 2021: https://arxiv.org/abs/2103.00020 — conceptual reference for image/text similarity; this repository does not reproduce its pretraining.
5. Reimers and Gurevych, *Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation*, 2020: https://arxiv.org/abs/2004.09813 — background for the multilingual text model.
6. PyTorch documentation: https://pytorch.org/docs/stable/
7. scikit-learn evaluation metrics: https://scikit-learn.org/stable/modules/model_evaluation.html

The package contains newly written code. External models and datasets retain their own licenses and attribution requirements.
