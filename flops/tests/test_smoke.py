import torch
from src.models import build_model

def test_models_shape():
 for name in ['unet','radiounet','fno','ufno']:
  c={'name':name,'in_channels':5,'out_channels':1,'base_width':8,'width':8,'modes1':4,'modes2':4,'depth':2,'padding':3}
  y=build_model(c)(torch.randn(1,5,32,32)); assert y.shape==(1,1,32,32)
