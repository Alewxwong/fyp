import argparse,json
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data import RadioMapDataset
from src.models import build_model
from src.metrics import batch_sums,finalize
p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--split',default='test'); p.add_argument('--device',default='auto'); a=p.parse_args()
dev=torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else ('cpu' if a.device=='auto' else a.device)); ck=torch.load(a.checkpoint,map_location=dev,weights_only=False); c=ck['config']; d=c['data']
ds=RadioMapDataset(d['manifest'],d['root'],a.split,d['split_dir'],d['input_mode'],d['tx_sigma'],False); dl=DataLoader(ds,batch_size=c['training']['batch_size'],num_workers=d['num_workers']); model=build_model(c['model']).to(dev); model.load_state_dict(ck['model']); model.eval(); totals={'se':0.,'ae':0.,'n':0,'nse':0.,'den':0.}; rows=[]
with torch.no_grad():
 for z in tqdm(dl):
  pred=model(z['x'].to(dev)).clamp(0,1).cpu().numpy(); y=z['y'].numpy()
  for i in range(len(pred)):
   s=batch_sums(pred[i:i+1],y[i:i+1],d['min_db'],d['max_db']); m=finalize(s); rows.append({'map_id':z['map_id'][i],'tx_id':z['tx_id'][i],**m})
   for k in totals: totals[k]+=s[k]
res=finalize(totals); out=Path(a.checkpoint).parent; (out/f'{a.split}_metrics.json').write_text(json.dumps(res,indent=2)); pd.DataFrame(rows).to_csv(out/f'{a.split}_per_sample.csv',index=False); print(json.dumps(res,indent=2))
