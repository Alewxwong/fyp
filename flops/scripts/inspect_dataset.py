import argparse
from pathlib import Path
from collections import Counter
from PIL import Image
p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--limit',type=int,default=30); a=p.parse_args()
root=Path(a.root); files=list(root.rglob('*.png'))
print(f'PNG count: {len(files)}')
print('Top directories:')
for k,v in Counter(str(x.parent.relative_to(root)) for x in files).most_common(30): print(v,k)
print('Examples:')
for x in files[:a.limit]:
    try: print(x.relative_to(root),Image.open(x).size)
    except Exception as e: print(x,e)
