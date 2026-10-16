import argparse,random
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--out',default='data/splits'); p.add_argument('--seed',type=int,default=42); p.add_argument('--train',type=float,default=.70); p.add_argument('--val',type=float,default=.15); a=p.parse_args()
ids=sorted(pd.read_csv(a.manifest,dtype=str).map_id.unique()); random.Random(a.seed).shuffle(ids)
n=len(ids); nt=round(n*a.train); nv=round(n*a.val); groups={'train':ids[:nt],'val':ids[nt:nt+nv],'test':ids[nt+nv:]}
out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
for k,v in groups.items(): (out/f'{k}_maps.txt').write_text('\n'.join(v)+'\n'); print(k,len(v))
assert not (set(groups['train'])&set(groups['val']) or set(groups['train'])&set(groups['test']) or set(groups['val'])&set(groups['test']))
