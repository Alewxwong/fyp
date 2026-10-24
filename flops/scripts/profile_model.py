import argparse,time,torch
from src.utils import load_config
from src.models import build_model
p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--device',default='auto'); a=p.parse_args(); c=load_config(a.config); c['model']['in_channels']=2 if c['data']['input_mode']=='two' else 5
d=torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else ('cpu' if a.device=='auto' else a.device)); m=build_model(c['model']).to(d).eval(); x=torch.randn(1,c['model']['in_channels'],256,256,device=d)
print('parameters',sum(p.numel() for p in m.parameters() if p.requires_grad))
with torch.no_grad():
 for _ in range(10): m(x)
 if d.type=='cuda': torch.cuda.synchronize()
 t=time.perf_counter()
 for _ in range(50): m(x)
 if d.type=='cuda': torch.cuda.synchronize()
 print('mean_latency_ms',(time.perf_counter()-t)/50*1000)
try:
 from fvcore.nn import FlopCountAnalysis
 print('flops',FlopCountAnalysis(m,x).total())
except Exception as e: print('FLOP count unavailable:',e)
