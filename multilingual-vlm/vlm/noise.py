"""Deterministic single-corruption perturbations applied at native resolution."""
import hashlib
import io
import numpy as np
from PIL import Image, ImageFilter
KINDS=['gaussian','salt_pepper','blur','jpeg','occlusion']

def stable_seed(seed, sample_id, kind, severity):
    return int.from_bytes(hashlib.sha256(f'{seed}|{sample_id}|{kind}|{severity}'.encode()).digest()[:8],'little')

def corrupt(image,kind='clean',severity=0,seed=0):
    image=image.convert('RGB')
    if kind=='clean': return image.copy()
    if kind not in KINDS or severity not in [1,2,3]: raise ValueError('Unknown corruption or severity')
    rng=np.random.default_rng(seed); i=severity-1
    if kind=='blur': return image.filter(ImageFilter.GaussianBlur([.5,1.,1.5][i]))
    if kind=='jpeg':
        b=io.BytesIO(); image.save(b,format='JPEG',quality=[70,40,15][i]); b.seek(0)
        with Image.open(b) as im: return im.convert('RGB').copy()
    a=np.array(image).astype(np.float32)
    if kind=='gaussian': a+=rng.normal(0,[10,20,35][i],a.shape)
    elif kind=='salt_pepper':
        mask=rng.random(a.shape[:2]); rate=[.02,.05,.1][i]
        a[mask<rate/2]=0; a[(mask>=rate/2)&(mask<rate)]=255
    elif kind=='occlusion':
        h,w=a.shape[:2]; side=[.15,.25,.35][i]
        hh,ww=max(1,int(h*side)),max(1,int(w*side))
        y,x=rng.integers(0,h-hh+1),rng.integers(0,w-ww+1)
        a[y:y+hh,x:x+ww]=0
    return Image.fromarray(np.clip(a,0,255).astype(np.uint8))
