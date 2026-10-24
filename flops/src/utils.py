import os, random, yaml
from pathlib import Path
import numpy as np
import torch

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def deep_update(a, b):
    for k,v in b.items():
        if isinstance(v,dict) and isinstance(a.get(k),dict): deep_update(a[k],v)
        else: a[k]=v
    return a

def load_config(path):
    path=Path(path); cfg=yaml.safe_load(path.read_text())
    if 'base' in cfg:
        base=yaml.safe_load((path.parent/cfg.pop('base')).read_text())
        cfg=deep_update(base,cfg)
    return cfg

def atomic_save(obj,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); torch.save(obj,tmp); os.replace(tmp,path)
