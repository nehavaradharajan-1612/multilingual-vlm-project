import hashlib
import importlib.metadata
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
from .data import read_manifest,manifest_fingerprint
from .noise import KINDS,corrupt,stable_seed
from .model import VisionLanguageClassifier
from .text import LANGUAGES

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)

def tensor_image(im):
    im=im.resize((64,64),Image.Resampling.BILINEAR)
    return torch.from_numpy(np.array(im,dtype=np.float32).transpose(2,0,1)/127.5-1.)

class Images(Dataset):
    def __init__(self,df,seed,kind='clean',severity=0,augment=False):
        self.rows=df.to_dict('records'); self.seed=seed; self.kind=kind; self.severity=severity
        self.augment=augment; self.epoch=0
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(r['path']) as source: im=source.convert('RGB')
        kind,severity=self.kind,self.severity
        s=stable_seed(self.seed,r['sample_id'],kind,severity)
        if self.augment:
            s=stable_seed(self.seed,r['sample_id'],'train',self.epoch)
            rng=np.random.default_rng(s)
            kind=rng.choice(['clean']+KINDS); severity=int(rng.integers(1,3)) if kind!='clean' else 0
        im=corrupt(im,kind,severity,s)
        return tensor_image(im),int(r['label']),r['sample_id']

def collect(model,df,device,batch_size,seed,kind='clean',severity=0):
    model.eval(); actual=[]; ids=[]; prediction=[]
    loader=DataLoader(Images(df,seed,kind,severity),batch_size=batch_size,shuffle=False,num_workers=0)
    with torch.inference_mode():
        for images,y,names in loader:
            prediction.append(model(images.to(device)).argmax(-1).cpu().numpy())
            actual.extend(y.tolist()); ids.extend(names)
    return np.array(actual),np.concatenate(prediction),ids

def metrics(y,p):
    return {'accuracy':float(accuracy_score(y,p)), 'balanced_accuracy':float(balanced_accuracy_score(y,p)),
            'macro_f1':float(f1_score(y,p,labels=list(range(15)),average='macro',zero_division=0))}

def train(manifest,prototypes,out,epochs=30,batch_size=64,lr=.001,seed=42,device='cpu',augment=False,patience=7):
    seed_everything(seed)
    if epochs<1 or batch_size<1 or patience<1 or lr<=0: raise ValueError('Invalid training settings')
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    df=read_manifest(manifest); train_df=df[df.split=='train']; val=df[df.split=='val']
    for part in [train_df,val]:
        if set(part.label)!=set(range(15)): raise ValueError('Training/validation must include all 15 classes')
    proto_dir=Path(prototypes)
    vectors=np.load(proto_dir/'prototypes.npz',allow_pickle=False)['vectors']
    if not np.isfinite(vectors).all() or (np.linalg.norm(vectors,axis=-1)==0).any(): raise ValueError('Invalid prototypes')
    text_meta=json.loads((proto_dir/'metadata.json').read_text())
    if text_meta['languages'] != LANGUAGES:
        raise ValueError('Prototype language order does not match the classifier')
    model=VisionLanguageClassifier(torch.from_numpy(vectors)).to(device)
    ds=Images(train_df,seed,augment=augment)
    loader=DataLoader(ds,batch_size=batch_size,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(seed))
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001)
    history=[]; best=-1.; stale=0
    for epoch in range(1,epochs+1):
        ds.epoch=epoch; model.train(); loss_sum=0.
        for images,y,_ in loader:
            images,y=images.to(device),y.to(device)
            opt.zero_grad(set_to_none=True)
            logits=model(images)
            loss=sum(F.cross_entropy(logits[:,i],y) for i in range(3))/3
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
            loss_sum+=float(loss.detach())*len(y)
        actual,pred,_=collect(model,val,device,batch_size,seed)
        row={'epoch':epoch,'train_loss':loss_sum/len(ds)}
        row.update({f'val_accuracy_{lang}':float(np.mean(pred[:,i]==actual)) for i,lang in enumerate(LANGUAGES)})
        score=float(np.mean(pred==actual[:,None])); row['val_accuracy_mean']=score; history.append(row)
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        print(json.dumps(row),flush=True)
        if score>best:
            best=score; stale=0
            torch.save({'state_dict':model.state_dict(),'dimension':vectors.shape[-1], 'epoch':epoch,
                'best_val_accuracy_mean':best,'text_metadata':text_meta,
                'manifest_sha256':hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                'dataset_fingerprint':manifest_fingerprint(manifest),
                'seed':seed},out/'best.pt')
        else: stale+=1
        if stale>=patience: break
    settings=dict(epochs=epochs,batch_size=batch_size,learning_rate=lr,seed=seed,device=device,
        augmentation=augment,patience=patience,selection='mean clean validation accuracy across zh,en,es',
        training_samples=len(ds),validation_samples=len(val),synthetic=text_meta['synthetic'],
        versions={p:importlib.metadata.version(p) for p in ['torch','numpy','pandas','Pillow','scikit-learn']},
        prototype_sha256=hashlib.sha256((proto_dir/'prototypes.npz').read_bytes()).hexdigest())
    (out/'run.json').write_text(json.dumps(settings,indent=2))
    df.to_csv(out/'manifest_snapshot.csv',index=False)
    plot_history(pd.DataFrame(history),out)

def load(checkpoint,device):
    artifact=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=VisionLanguageClassifier(torch.zeros(3,15,artifact['dimension']))
    model.load_state_dict(artifact['state_dict']); model.to(device).eval()
    return model,artifact

def evaluate(manifest,checkpoint,out,device='cpu',batch_size=64,robust=False):
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    model,artifact=load(checkpoint,device); seed_everything(artifact['seed'])
    expected=artifact.get('dataset_fingerprint')
    matches=(manifest_fingerprint(manifest)==expected) if expected else (hashlib.sha256(Path(manifest).read_bytes()).hexdigest()==artifact['manifest_sha256'])
    if not matches:
        raise ValueError('Dataset identities, labels or splits differ from training')
    df=read_manifest(manifest); df=df[df.split=='test']
    if set(df.label)!=set(range(15)): raise ValueError('Test must include all classes')
    conditions=[('clean',0)]+([(k,s) for k in KINDS for s in [1,2,3]] if robust else [])
    rows=[]; predictions=[]
    for kind,severity in conditions:
        actual,pred,ids=collect(model,df,device,batch_size,artifact['seed'],kind,severity)
        for i,lang in enumerate(LANGUAGES):
            rows.append(dict(language=lang,corruption=kind,severity=severity,**metrics(actual,pred[:,i])))
            predictions.extend(dict(sample_id=s,language=lang,corruption=kind,severity=severity,actual=int(y),predicted=int(p)) for s,y,p in zip(ids,actual,pred[:,i]))
            if kind=='clean':
                pd.DataFrame(confusion_matrix(actual,pred[:,i],labels=list(range(15)))).to_csv(out/f'confusion_{lang}.csv',index=False)
        print(f'Evaluated {kind} severity {severity}',flush=True)
    result=pd.DataFrame(rows)
    clean=result[result.corruption=='clean'].set_index('language').accuracy.to_dict()
    result['accuracy_drop_pp']=[100*(clean[r.language]-r.accuracy) for r in result.itertuples()]
    result.to_csv(out/'metrics.csv',index=False); pd.DataFrame(predictions).to_csv(out/'predictions.csv',index=False)
    (out/'evaluation.json').write_text(json.dumps({'seed':artifact['seed'],'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        'test_samples':len(df),'synthetic':artifact['text_metadata']['synthetic'],'noise_replicates':1},indent=2))
    plot_results(result,out)
    plot_confusions(out)

def plot_results(result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
    for ax,lang in zip(axes,LANGUAGES):
        part=result[result.language==lang]; clean=float(part[part.corruption=='clean'].accuracy.iloc[0])
        ax.axhline(clean,color='black',linestyle='--',label='clean')
        for kind in KINDS:
            p=part[part.corruption==kind]
            if len(p): ax.plot(p.severity,p.accuracy,'o-',label=kind)
        ax.set(title=lang,xlabel='Corruption severity',ylim=(0,1.02),xticks=[1,2,3])
    axes[0].set_ylabel('Test accuracy'); axes[-1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out/'robustness.png',dpi=180); plt.close(fig)

def predict(checkpoint,image,device='cpu'):
    model,artifact=load(checkpoint,device)
    with Image.open(image) as im: x=tensor_image(im.convert('RGB')).unsqueeze(0).to(device)
    with torch.inference_mode(): p=model(x).softmax(-1)[0].cpu().numpy()
    from .data import CHARACTERS
    return {'synthetic':artifact['text_metadata']['synthetic'],'predictions':{lang:{'class_id':int(p[i].argmax()),
        'character':CHARACTERS[int(p[i].argmax())],'probabilities':p[i].tolist()} for i,lang in enumerate(LANGUAGES)}}


def plot_history(history,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot(history.epoch,history.train_loss,'o-')
    axes[0].set(xlabel='Epoch',ylabel='Training cross-entropy')
    for lang in LANGUAGES:
        axes[1].plot(history.epoch,history[f'val_accuracy_{lang}'],label=lang)
    axes[1].set(xlabel='Epoch',ylabel='Clean validation accuracy',ylim=(0,1.02))
    axes[1].legend(); fig.tight_layout()
    fig.savefig(out/'learning_curve.png',dpi=180); plt.close(fig)

def plot_confusions(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,lang in zip(axes,LANGUAGES):
        a=pd.read_csv(out/f'confusion_{lang}.csv').to_numpy()
        ax.imshow(a,cmap='Blues')
        ax.set(title=lang,xlabel='Predicted class ID',ylabel='True class ID',xticks=range(15),yticks=range(15))
        ax.tick_params(labelsize=6)
    fig.tight_layout(); fig.savefig(out/'confusion_matrices.png',dpi=180); plt.close(fig)
