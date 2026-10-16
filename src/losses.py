import torch
import torch.nn.functional as F

def gradient_loss(p,y):
    px=p[:,:,:,1:]-p[:,:,:,:-1]; yx=y[:,:,:,1:]-y[:,:,:,:-1]
    py=p[:,:,1:,:]-p[:,:,:-1,:]; yy=y[:,:,1:,:]-y[:,:,:-1,:]
    return F.l1_loss(px,yx)+F.l1_loss(py,yy)

def make_loss(name='l1',gradient_weight=0.0):
    def loss(p,y):
        base=F.l1_loss(p,y) if name=='l1' else F.mse_loss(p,y)
        return base+gradient_weight*gradient_loss(p,y)
    return loss
