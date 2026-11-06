import json
from pathlib import Path
import numpy as np
from .data import prompts
MODEL='sentence-transformers/clip-ViT-B-32-multilingual-v1'
LANGUAGES=['zh','en','es']

def build(out,revision=None,demo=False):
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    if demo:
        vectors=np.repeat(np.eye(15,dtype=np.float32)[None],3,axis=0)
        model_name='SYNTHETIC_IDENTITY_NOT_LANGUAGE_EMBEDDINGS'
    else:
        import torch
        torch.set_num_threads(2)
        from sentence_transformers import SentenceTransformer
        model=SentenceTransformer(MODEL,revision=revision,device='cpu',trust_remote_code=False)
        vectors=np.stack([model.encode(prompts()[lang],normalize_embeddings=True) for lang in LANGUAGES])
        model_name=MODEL
    np.savez_compressed(out/'prototypes.npz',vectors=vectors)
    (out/'metadata.json').write_text(json.dumps({'model':model_name,'revision_requested':revision,
        'languages':LANGUAGES,'prompts':prompts(),'synthetic':demo},ensure_ascii=False,indent=2))
