import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

CHARACTERS = list('零一二三四五六七八九十百千万亿')
ENGLISH = ['zero','one','two','three','four','five','six','seven','eight','nine','ten','one hundred','one thousand','ten thousand','one hundred million']
SPANISH = ['cero','uno','dos','tres','cuatro','cinco','seis','siete','ocho','nueve','diez','cien','mil','diez mil','cien millones']
VALUES = [0,1,2,3,4,5,6,7,8,9,10,100,1000,10000,100000000]

def prompts():
    return {'zh': [f'手写汉字“{c}”。' for c in CHARACTERS],
            'en': [f'The handwritten Chinese numeral for {c}.' for c in ENGLISH],
            'es': [f'El numeral chino manuscrito que significa {c}.' for c in SPANISH]}

def split_writers(df, seed):
    writers = np.array(sorted(df.writer.unique()))
    if len(writers) < 10:
        raise ValueError('Need at least 10 writers for writer-disjoint splits')
    np.random.default_rng(seed).shuffle(writers)
    n = len(writers); a, b = int(.7*n), int(.85*n)
    membership = {w: ('train' if i<a else 'val' if i<b else 'test') for i,w in enumerate(writers)}
    df = df.copy(); df['split'] = df.writer.map(membership)
    for split in ['train','val','test']:
        if set(df.loc[df.split == split,'label']) != set(range(15)):
            raise ValueError(f'{split} is missing classes')
    return df

def prepare(root, out, seed=42):
    root, out = Path(root).resolve(), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    csvs = list(root.rglob('chinese_mnist.csv'))
    if len(csvs) != 1:
        raise ValueError('Expected exactly one chinese_mnist.csv under root')
    raw = pd.read_csv(csvs[0])
    required = {'suite_id','sample_id','code','character','value'}
    if not required.issubset(raw):
        raise ValueError(f'Missing metadata columns: {required-set(raw)}')
    images = {}
    for p in root.rglob('input_*.jpg'):
        if p.name in images:
            raise ValueError(f'Duplicate image filename: {p.name}')
        images[p.name] = p
    rows=[]; seen={}
    for r in raw.itertuples():
        code=int(r.code)
        if not 1 <= code <= 15 or str(r.character) != CHARACTERS[code-1] or int(r.value) != VALUES[code-1]:
            raise ValueError('Dataset code/character/value mapping does not match expected release')
        name=f'input_{int(r.suite_id)}_{int(r.sample_id)}_{code}.jpg'
        p=images[name]
        with Image.open(p) as im:
            rgb=im.convert('RGB'); rgb.load()
            digest=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
        if digest in seen:
            raise ValueError(f'Duplicate decoded image: {name} and {seen[digest]}; curate explicitly')
        seen[digest]=name
        rows.append(dict(sample_id=p.stem,path=str(p),label=code-1,writer=str(r.suite_id),sha256=digest))
    df=split_writers(pd.DataFrame(rows),seed)
    if df.sample_id.duplicated().any(): raise ValueError('Duplicate sample IDs')
    df.to_csv(out/'manifest.csv',index=False)
    (out/'data_card.json').write_text(json.dumps({'source':'gpreda/chinese-mnist','seed':seed,
        'samples':len(df),'writers':df.writer.nunique(),'counts':df.groupby(['split','label']).size().unstack(fill_value=0).to_dict(),
        'classes':CHARACTERS,'prompts':prompts()},ensure_ascii=False,indent=2))

def read_manifest(path):
    path=Path(path).resolve()
    df=pd.read_csv(path,dtype={'sample_id':str,'writer':str})
    if not {'sample_id','path','label','writer','split'}.issubset(df): raise ValueError('Incomplete manifest')
    if df.isna().any().any() or df.sample_id.duplicated().any(): raise ValueError('Missing values or duplicate IDs')
    if not set(df.split).issubset({'train','val','test'}): raise ValueError('Invalid split')
    if df.groupby('writer').split.nunique().max()!=1: raise ValueError('Writer leakage across splits')
    if not np.equal(df.label,df.label.astype(int)).all() or not df.label.between(0,14).all(): raise ValueError('Invalid labels')
    if 'sha256' in df and df.sha256.duplicated().any(): raise ValueError('Duplicate image hashes')
    df['path']=[str((path.parent/p).resolve()) for p in df.path]
    return df

def demo(out,seed=42):
    """Synthetic geometric patterns, NOT Chinese handwriting or validation data."""
    out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for writer in range(20):
        for label in range(15):
            im=Image.new('RGB',(64,64),'black'); d=ImageDraw.Draw(im)
            x=5+(label%5)*11; y=7+(label//5)*17
            d.rectangle([x,y,x+7,y+8],fill='white')
            d.point((writer+1,60),fill=(writer*10,)*3)
            name=f'synthetic_{writer}_{label}'
            im.save(out/f'{name}.png')
            rows.append(dict(sample_id=name,path=str(out/f'{name}.png'),label=label,writer=str(writer)))
    split_writers(pd.DataFrame(rows),seed).to_csv(out/'manifest.csv',index=False)


def manifest_fingerprint(path):
    """Portable identity: labels, writer groups, split IDs and image hashes, not paths."""
    df=read_manifest(path)
    cols=['sample_id','label','writer','split']+(['sha256'] if 'sha256' in df else [])
    payload=df[cols].sort_values('sample_id').to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()
