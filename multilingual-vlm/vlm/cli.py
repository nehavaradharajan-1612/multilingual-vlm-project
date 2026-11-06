import argparse
import json
from . import data,text

def main():
    p=argparse.ArgumentParser(description='Multilingual vision-language classification')
    sub=p.add_subparsers(dest='cmd',required=True)
    d=sub.add_parser('download')
    d=sub.add_parser('prepare'); d.add_argument('--root',required=True); d.add_argument('--out',required=True); d.add_argument('--seed',type=int,default=42)
    d=sub.add_parser('demo'); d.add_argument('--out',required=True)
    d=sub.add_parser('text'); d.add_argument('--out',required=True); d.add_argument('--revision'); d.add_argument('--demo',action='store_true')
    d=sub.add_parser('train'); d.add_argument('--manifest',required=True); d.add_argument('--prototypes',required=True); d.add_argument('--out',required=True)
    d.add_argument('--epochs',type=int,default=30); d.add_argument('--batch-size',type=int,default=64); d.add_argument('--lr',type=float,default=.001)
    d.add_argument('--seed',type=int,default=42); d.add_argument('--device',default='cpu'); d.add_argument('--augment',action='store_true'); d.add_argument('--patience',type=int,default=7)
    d=sub.add_parser('evaluate'); d.add_argument('--manifest',required=True); d.add_argument('--checkpoint',required=True); d.add_argument('--out',required=True)
    d.add_argument('--device',default='cpu'); d.add_argument('--batch-size',type=int,default=64); d.add_argument('--robust',action='store_true')
    d=sub.add_parser('predict'); d.add_argument('--checkpoint',required=True); d.add_argument('--image',required=True); d.add_argument('--device',default='cpu')
    a=p.parse_args()
    if a.cmd=='download':
        import kagglehub
        print(kagglehub.dataset_download('gpreda/chinese-mnist'))
    elif a.cmd=='prepare': data.prepare(a.root,a.out,a.seed)
    elif a.cmd=='demo': data.demo(a.out)
    elif a.cmd=='text': text.build(a.out,a.revision,a.demo)
    else:
        from . import engine
        if a.cmd=='train': engine.train(a.manifest,a.prototypes,a.out,a.epochs,a.batch_size,a.lr,a.seed,a.device,a.augment,a.patience)
        elif a.cmd=='evaluate': engine.evaluate(a.manifest,a.checkpoint,a.out,a.device,a.batch_size,a.robust)
        else: print(json.dumps(engine.predict(a.checkpoint,a.image,a.device),ensure_ascii=False,indent=2))
if __name__=='__main__': main()
