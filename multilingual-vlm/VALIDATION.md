# Validation and measured results

Validation date: 6 November 2026 (Asia/Kolkata).

## Real dataset and model

The Chinese MNIST archive was downloaded successfully from Kaggle. All 15,000 images decoded, all metadata label mappings passed validation, and no duplicate decoded-image hashes were found. The seeded writer split contains 10,500 training images from 70 writers, 2,250 validation images from 15 writers, and 2,250 test images from 15 writers.

The multilingual text encoder was downloaded and executed successfully. Its downloaded model revision is `58edf8cada9e398793dca955574a48cbb7f18be2`. The real cached vectors are included in `artifacts/text/`.

## Training experiment

One predeclared five-epoch experiment was run on CPU with seed 42, batch size 128, learning rate 0.001, AdamW, and noise augmentation. All 10,500 training images were used in every epoch. The image encoder was trained from scratch; the language encoder remained frozen.

The selected checkpoint is epoch 5, with **86.89% mean clean validation accuracy** across the three prompt languages:

| Prompt language | Validation accuracy |
|---|---:|
| Chinese | 85.78% |
| English | 91.73% |
| Spanish | 83.16% |
| Mean across languages | 86.89% |

These are actual recorded results, not a target or an estimate. The supplied résumé's 96% validation figure was **not reproduced**. The default 30-epoch configuration was not run, and further training has no guaranteed outcome.

`artifacts/best.pt` contains the actual selected model. The export adds a portable dataset fingerprint and downloaded text-model revision metadata; this changes no learned weights or predictions. `evidence/real_run/history.csv` and `run.json` preserve the measured history and training settings.

## Software checks

- Nine automated tests passed, including writer leakage rejection, deterministic corruptions, prompt alignment, gradient flow with frozen text embeddings, checkpoint round-trip equality, and portable dataset identity checks.
- All Python modules passed compilation; notebook code cells passed syntax checks.
- The synthetic 300-image smoke test completed training, checkpoint loading, and all 48 evaluation conditions. Its 45 test images appeared in every condition. Synthetic identity prototypes are not language embeddings, and those results are not scientific evidence about Chinese images.
- Single-image inference was exercised on both synthetic and real checkpoints; the real-model test correctly classified a source example of 零 under all three prompt languages. This individual example is a functional check, not an additional accuracy estimate.
- The package installed successfully in editable mode. The notebook was validated structurally but was not executed end-to-end as a notebook; its underlying modules were exercised directly.

## Limits of this experiment

This is one seed, one train/validation/test partition, and one five-epoch model. No separate clean-only baseline, conventional image classifier, full CLIP fine-tuning, multiple-seed uncertainty study, or unseen-script benchmark was run. The three prompt languages share the same Chinese test images. The corruption study is controlled synthetic degradation, not a test on naturally collected noisy multilingual scenes.

## Held-out test evaluation

All 2,250 held-out test images were evaluated in every language/corruption condition: 48 metric rows and 108,000 individual prediction rows. No test result was used to change training or select the checkpoint.

| Prompt language | Clean test accuracy | Clean macro F1 | Mean accuracy over 15 corrupted conditions |
|---|---:|---:|---:|
| zh | 87.69% | 0.8580 | 81.29% |
| en | 93.56% | 0.9356 | 85.95% |
| es | 84.67% | 0.8378 | 79.56% |

Mean clean test accuracy across the three prompt languages: **88.64%**. The corrupted-condition average equally weights the five corruption types and three severities; it is a project summary, not an official benchmark score.

Complete predictions, per-condition metrics, confusion matrices, plots, and checkpoint checksum are in `evidence/real_test/`.
