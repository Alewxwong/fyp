import argparse,re
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser(description='Pair building, Tx, and target PNGs using named regex groups map and tx.')
p.add_argument('--root',required=True); p.add_argument('--building-glob',required=True); p.add_argument('--tx-glob',required=True); p.add_argument('--target-glob',required=True)
p.add_argument('--building-regex',default=r'(?P<map>\d+)'); p.add_argument('--tx-regex',default=r'(?P<map>\d+)[^0-9]+(?P<tx>\d+)'); p.add_argument('--target-regex',default=r'(?P<map>\d+)[^0-9]+(?P<tx>\d+)'); p.add_argument('--out',required=True)
a=p.parse_args(); root=Path(a.root)
def key(path,regex,need_tx):
    m=re.search(regex,path.stem)
    if not m: return None
    return (m.group('map').lstrip('0') or '0',m.group('tx').lstrip('0') or '0') if need_tx else (m.group('map').lstrip('0') or '0')
b={key(x,a.building_regex,False):x for x in root.glob(a.building_glob) if key(x,a.building_regex,False)}
t={key(x,a.tx_regex,True):x for x in root.glob(a.tx_glob) if key(x,a.tx_regex,True)}
y={key(x,a.target_regex,True):x for x in root.glob(a.target_glob) if key(x,a.target_regex,True)}
rows=[]
for k in sorted(set(t)&set(y)):
    if k[0] in b: rows.append({'map_id':k[0],'tx_id':k[1],'building_path':b[k[0]].relative_to(root),'tx_path':t[k].relative_to(root),'target_path':y[k].relative_to(root)})
out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out,index=False)
print(f'buildings={len(b)} tx={len(t)} targets={len(y)} paired={len(rows)} -> {out}')
if not rows: raise SystemExit('No pairs. Inspect filenames and change glob/regex arguments.')
