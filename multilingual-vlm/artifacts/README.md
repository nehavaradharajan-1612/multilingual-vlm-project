# Included experimental artifacts

`text/` contains real multilingual embeddings of the 45 class prompts, produced by the documented frozen text model. They let you train the image model without downloading the large language model again:

```bash
vlm train --manifest data/prepared/manifest.csv --prototypes artifacts/text --out runs/local --augment
```

`best.pt`, when present, is the selected checkpoint from the delivery experiment. It is a five-epoch experimental model, not a certified 96%-accuracy model. The checkpoint contains its text prototypes, so `vlm predict` works without separately loading the text encoder.

Read `VALIDATION.md` for its measured performance and `evidence/real_run/` for the exact experiment records. Download and prepare the original dataset with seed 42 before evaluating it yourself. The evaluation fingerprint permits different filesystem paths while requiring the same image identifiers, decoded-image hashes, targets, writer assignments and splits.
