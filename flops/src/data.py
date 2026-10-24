from pathlib import Path
import math, random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

def read_gray(path):
    return np.asarray(Image.open(path).convert('L'), dtype=np.float32)/255.0

def tx_coordinates(tx):
    y,x=np.unravel_index(int(np.argmax(tx)),tx.shape)
    return x,y

def gaussian_tx(h,w,x,y,sigma):
    yy,xx=np.mgrid[0:h,0:w]
    g=np.exp(-((xx-x)**2+(yy-y)**2)/(2*sigma*sigma))
    return g.astype(np.float32)

def build_channels(building,tx,input_mode='five',sigma=2.0):
    h,w=building.shape; x,y=tx_coordinates(tx)
    txh=gaussian_tx(h,w,x,y,sigma) if sigma>0 else tx
    if input_mode=='two': return np.stack([building,txh]).astype(np.float32)
    yy,xx=np.mgrid[0:h,0:w]
    dx=(xx-x)/max(w-1,1); dy=(yy-y)/max(h-1,1)
    dist=np.sqrt((xx-x)**2+(yy-y)**2)/math.sqrt((w-1)**2+(h-1)**2)
    return np.stack([building,txh,dist,dx,dy]).astype(np.float32)

def rotate_flip(arrs,k,hflip,vflip):
    out=[]
    for a in arrs:
        a=np.rot90(a,k,axes=(-2,-1))
        if hflip: a=np.flip(a,axis=-1)
        if vflip: a=np.flip(a,axis=-2)
        out.append(np.ascontiguousarray(a))
    return out

class RadioMapDataset(Dataset):
    def __init__(self,manifest,root,split,split_dir,input_mode='five',sigma=2.0,augment=False):
        df=pd.read_csv(manifest,dtype={'map_id':str,'tx_id':str})
        ids=set(x.strip() for x in Path(split_dir,f'{split}_maps.txt').read_text().splitlines() if x.strip())
        self.df=df[df.map_id.astype(str).isin(ids)].reset_index(drop=True)
        if len(self.df)==0: raise RuntimeError(f'No samples for {split}; check map IDs and split files')
        self.root=Path(root); self.mode=input_mode; self.sigma=sigma; self.augment=augment
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        b=read_gray(self.root/r.building_path)
        t=read_gray(self.root/r.tx_path)
        y=read_gray(self.root/r.target_path)[None]
        if self.augment:
            k=random.randrange(4); hf=random.random()<.5; vf=random.random()<.5
            b,t,y=rotate_flip([b,t,y],k,hf,vf)
        x=build_channels(b,t,self.mode,self.sigma)
        return {'x':torch.from_numpy(x),'y':torch.from_numpy(y),
                'map_id':str(r.map_id),'tx_id':str(r.tx_id)}
