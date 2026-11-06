import numpy as np
import pandas as pd
from PIL import Image
import pytest
from vlm.noise import corrupt,stable_seed,KINDS
from vlm.data import demo,read_manifest,prompts,CHARACTERS

def test_corruptions_deterministic():
    im=Image.fromarray(np.arange(64*64*3,dtype=np.uint8).reshape(64,64,3))
    for kind in KINDS:
        for severity in [1,2,3]:
            a=corrupt(im,kind,severity,42); b=corrupt(im,kind,severity,42)
            assert a.size==im.size and a.mode=='RGB'
            assert np.array_equal(np.asarray(a),np.asarray(b))
    assert np.array_equal(np.asarray(im),np.asarray(corrupt(im)))

def test_bad_noise():
    with pytest.raises(ValueError): corrupt(Image.new('RGB',(64,64)),'bad',1)

def test_seed_independent_of_order():
    assert stable_seed(42,'a','gaussian',1)==stable_seed(42,'a','gaussian',1)
    assert stable_seed(42,'a','gaussian',1)!=stable_seed(42,'b','gaussian',1)

def test_writer_split(tmp_path):
    demo(tmp_path/'data'); df=read_manifest(tmp_path/'data/manifest.csv')
    assert df.groupby('writer').split.nunique().max()==1
    assert df.groupby('split').label.nunique().eq(15).all()
    assert df.groupby('split').size().to_dict()=={'train':210,'val':45,'test':45}

def test_leakage_rejected(tmp_path):
    demo(tmp_path/'data'); p=tmp_path/'data/manifest.csv'; df=pd.read_csv(p)
    df.loc[0,'split']='test' if df.loc[1,'split']!='test' else 'train'; df.to_csv(p,index=False)
    with pytest.raises(ValueError,match='Writer leakage'): read_manifest(p)

def test_prompt_alignment():
    assert len(CHARACTERS)==15
    assert set(prompts())=={'zh','en','es'}
    assert all(len(p)==15 and len(set(p))==15 for p in prompts().values())

def test_gradient_and_frozen_text():
    torch=pytest.importorskip('torch')
    from vlm.model import VisionLanguageClassifier
    torch.set_num_threads(2)
    model=VisionLanguageClassifier(torch.randn(3,15,32))
    logits=model(torch.randn(2,3,64,64))
    assert logits.shape==(2,3,15)
    loss=torch.nn.functional.cross_entropy(logits[:,0],torch.tensor([1,2])); loss.backward()
    assert model.vision[0].weight.grad is not None
    assert model.prototypes.grad is None
    assert torch.isfinite(logits).all()

def test_checkpoint_roundtrip(tmp_path):
    torch=pytest.importorskip('torch')
    from vlm.model import VisionLanguageClassifier
    model=VisionLanguageClassifier(torch.randn(3,15,32)).eval()
    torch.save(model.state_dict(),tmp_path/'state.pt')
    other=VisionLanguageClassifier(torch.zeros(3,15,32)).eval()
    other.load_state_dict(torch.load(tmp_path/'state.pt',weights_only=True))
    x=torch.randn(2,3,64,64)
    with torch.inference_mode(): assert torch.equal(model(x),other(x))

def test_portable_fingerprint(tmp_path):
    from vlm.data import manifest_fingerprint
    demo(tmp_path/'data'); p=tmp_path/'data/manifest.csv'
    original=manifest_fingerprint(p)
    df=pd.read_csv(p); df['path']='/relocated/'+df.sample_id+'.png'; df.to_csv(p,index=False)
    assert manifest_fingerprint(p)==original
    df.loc[0,'label']=(df.loc[0,'label']+1)%15;df.to_csv(p,index=False)
    assert manifest_fingerprint(p)!=original
