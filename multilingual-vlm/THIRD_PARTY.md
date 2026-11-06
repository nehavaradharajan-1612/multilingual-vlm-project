# External data and model attribution

- **Chinese MNIST**, distributed by Gabriel Preda on Kaggle: https://www.kaggle.com/datasets/gpreda/chinese-mnist . Original images and CSV are not redistributed. Obtain the source through the publisher and observe the current dataset terms. This package includes experiment split identifiers and checksums derived from the downloaded release.
- **sentence-transformers/clip-ViT-B-32-multilingual-v1**: https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1 . The model card lists Apache-2.0. Cached prompt embeddings, if included, were produced from the 45 prompts listed in README.md and metadata.json. Original pretrained language-model weights are not bundled.
- **PyTorch, Sentence Transformers, NumPy, pandas, Pillow, scikit-learn, Matplotlib** retain their respective licenses. They are installed separately rather than redistributed here.

The image classifier is newly trained project work. Its checkpoint is not an original pretrained CLIP image model and must not be attributed as one.
