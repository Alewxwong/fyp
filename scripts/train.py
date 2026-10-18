import argparse,json,time
from itertools import islice
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.utils import load_config,seed_everything,atomic_save
from src.data import RadioMapDataset
from src.models import build_model
from src.losses import make_loss
from src.metrics import batch_sums,finalize

def loader(cfg,split):
    d=cfg['data']; ds=RadioMapDataset(d['manifest'],d['root'],split,d['split_dir'],d['input_mode'],d['tx_sigma'],d['augment'] and split=='train')
    return DataLoader(ds,batch_size=cfg['training']['batch_size'],shuffle=split=='train',num_workers=d['num_workers'],pin_memory=d['pin_memory'],persistent_workers=d['num_workers']>0)

def run(model,dl,device,lossfn,opt,scaler,cfg,train,max_batches=None,phase='Train',epoch=1,epochs=1):
    model.train(train)
    total=0.
    processed=0
    sums={'se':0.,'ae':0.,'n':0,'nse':0.,'den':0.}
    batches=islice(dl,max_batches) if max_batches is not None else dl
    shown_total=min(len(dl),max_batches) if max_batches is not None else len(dl)
    ctx=torch.enable_grad if train else torch.no_grad
    with ctx():
      for z in tqdm(
        batches,
        total=shown_total,
        desc=f"Epoch {epoch:02d}/{epochs:02d} {phase}",
        unit="batch",
        leave=True,
        dynamic_ncols=True,
      ):
        x=z['x'].to(device,non_blocking=True); y=z['y'].to(device,non_blocking=True)
        if train: opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,enabled=scaler.is_enabled()): p=model(x); loss=lossfn(p,y)
        if train:
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['training']['grad_clip']); scaler.step(opt); scaler.update()
        processed+=x.size(0); total+=loss.item()*x.size(0); s=batch_sums(p.detach().float().cpu().numpy(),y.cpu().numpy(),cfg['data']['min_db'],cfg['data']['max_db'])
        for k in sums: sums[k]+=s[k]
    return total/max(processed,1),finalize(sums)

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',required=True)
 p.add_argument('--device',default='auto')
 p.add_argument('--max-epochs',type=int)
 p.add_argument('--max-train-batches',type=int,default=None)
 p.add_argument('--max-val-batches',type=int,default=None)
 a=p.parse_args(); cfg=load_config(a.config); seed_everything(cfg['seed'])
 device=torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else ('cpu' if a.device=='auto' else a.device)); print('device',device)
 cin=2 if cfg['data']['input_mode']=='two' else 5; cfg['model']['in_channels']=cin
 model=build_model(cfg['model']).to(device); print('parameters',sum(p.numel() for p in model.parameters() if p.requires_grad))
 tr,va=loader(cfg,'train'),loader(cfg,'val'); opt=torch.optim.AdamW(model.parameters(),lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay']); epochs=a.max_epochs or cfg['training']['epochs']; sch=torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=epochs,
        eta_min=cfg['training'].get('min_lr', 1e-6)
    ); lossfn=make_loss(cfg['training']['loss'],cfg['training']['gradient_weight']); scaler=torch.amp.GradScaler('cuda',enabled=cfg['training']['amp'] and device.type=='cuda')
 run_dir=Path(cfg['output_dir'])/cfg['model']['name']/f"seed_{cfg['seed']}"; run_dir.mkdir(parents=True,exist_ok=True); writer=SummaryWriter(run_dir/'tb'); best=float('inf'); bad=0
 for ep in range(1,epochs+1):
    tl,tm=run(
        model,tr,device,lossfn,opt,scaler,cfg,True,
        max_batches=a.max_train_batches,
        phase='Train',epoch=ep,epochs=epochs
    )
    vl,vm=run(
        model,va,device,lossfn,opt,scaler,cfg,False,
        max_batches=a.max_val_batches,
        phase='Valid',epoch=ep,epochs=epochs
    )
    sch.step(); print(f"epoch {ep}: train={tl:.5f} val={vl:.5f} rmse={vm['rmse_db']:.3f}dB")
    writer.add_scalars('loss',{'train':tl,'val':vl},ep); writer.add_scalar('val/rmse_db',vm['rmse_db'],ep)
    state={'epoch':ep,'model':model.state_dict(),'optimizer':opt.state_dict(),'config':cfg,'val':vm}
    atomic_save(state,run_dir/'last.pt')
    if vl<best: best=vl; bad=0; atomic_save(state,run_dir/'best.pt')
    else: bad+=1
    if bad>=cfg['training']['patience']: print('early stopping'); break
 writer.close()
if __name__=='__main__': main()
