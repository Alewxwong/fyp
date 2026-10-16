import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self,ci,co):
        super().__init__(); g=min(8,co)
        while co%g: g-=1
        self.net=nn.Sequential(nn.Conv2d(ci,co,3,padding=1,bias=False),nn.GroupNorm(g,co),nn.GELU(),nn.Conv2d(co,co,3,padding=1,bias=False),nn.GroupNorm(g,co),nn.GELU())
    def forward(self,x): return self.net(x)

class UNet(nn.Module):
    def __init__(self,cin=5,cout=1,b=32):
        super().__init__(); cs=[b,b*2,b*4,b*8]
        self.enc=nn.ModuleList(); p=cin
        for c in cs: self.enc.append(ConvBlock(p,c)); p=c
        self.mid=ConvBlock(cs[-1],cs[-1]*2)
        self.up=nn.ModuleList([ConvBlock(cs[-1]*2+cs[-1],cs[-1]),ConvBlock(cs[-1]+cs[-2],cs[-2]),ConvBlock(cs[-2]+cs[-3],cs[-3]),ConvBlock(cs[-3]+cs[-4],cs[-4])])
        self.out=nn.Conv2d(b,cout,1)
    def forward(self,x):
        skips=[]
        for e in self.enc: x=e(x); skips.append(x); x=F.max_pool2d(x,2)
        x=self.mid(x)
        for block,s in zip(self.up,reversed(skips)):
            x=F.interpolate(x,size=s.shape[-2:],mode='bilinear',align_corners=False); x=block(torch.cat([x,s],1))
        return self.out(x)

class RadioUNet(nn.Module):
    # W-shaped two-stage radio-map CNN: coarse prediction then refinement.
    def __init__(self,cin=5,cout=1,b=24):
        super().__init__(); self.coarse=UNet(cin,cout,b); self.refine=UNet(cin+cout,cout,b)
    def forward(self,x):
        c=self.coarse(x); return self.refine(torch.cat([x,c],1))

class SpectralConv2d(nn.Module):
    def __init__(self,ci,co,m1,m2):
        super().__init__(); self.ci=ci; self.co=co; self.m1=m1; self.m2=m2
        scale=1/(ci*co); shape=(ci,co,m1,m2)
        self.w1=nn.Parameter(scale*torch.randn(*shape,dtype=torch.cfloat)); self.w2=nn.Parameter(scale*torch.randn(*shape,dtype=torch.cfloat))
    def compl_mul(self,x,w): return torch.einsum('bixy,ioxy->boxy',x,w)
    def forward(self,x):
        b,_,h,w=x.shape; xf=torch.fft.rfft2(x,norm='ortho')
        m1=min(self.m1,h//2); m2=min(self.m2,xf.shape[-1])
        out=torch.zeros(b,self.co,h,xf.shape[-1],device=x.device,dtype=torch.cfloat)
        out[:,:,:m1,:m2]=self.compl_mul(xf[:,:,:m1,:m2],self.w1[:,:,:m1,:m2])
        out[:,:,-m1:,:m2]=self.compl_mul(xf[:,:,-m1:,:m2],self.w2[:,:,:m1,:m2])
        return torch.fft.irfft2(out,s=(h,w),norm='ortho')

class MiniUNet(nn.Module):
    def __init__(self,c):
        super().__init__(); self.a=ConvBlock(c,c); self.b=ConvBlock(c,c*2); self.c=ConvBlock(c*2+c,c)
    def forward(self,x):
        s=self.a(x); z=self.b(F.avg_pool2d(s,2)); z=F.interpolate(z,size=s.shape[-2:],mode='bilinear',align_corners=False); return self.c(torch.cat([z,s],1))

class FourierBlock(nn.Module):
    def __init__(self,c,m1,m2,use_unet=False):
        super().__init__(); self.spec=SpectralConv2d(c,c,m1,m2); self.local=nn.Conv2d(c,c,1); self.u=MiniUNet(c) if use_unet else None; self.norm=nn.GroupNorm(1,c)
    def forward(self,x):
        z=self.spec(x)+self.local(x)
        if self.u is not None: z=z+self.u(x)
        return F.gelu(self.norm(z))

class FNO2d(nn.Module):
    def __init__(self,cin=5,cout=1,width=48,m1=20,m2=20,depth=4,padding=9,use_unet=False):
        super().__init__(); self.padding=padding; self.lift=nn.Conv2d(cin,width,1)
        self.blocks=nn.ModuleList([FourierBlock(width,m1,m2,use_unet and i>=depth//2) for i in range(depth)])
        self.proj=nn.Sequential(nn.Conv2d(width,128,1),nn.GELU(),nn.Conv2d(128,cout,1))
    def forward(self,x):
        x=self.lift(x); x=F.pad(x,[0,self.padding,0,self.padding])
        for b in self.blocks: x=b(x)
        if self.padding: x=x[...,:-self.padding,:-self.padding]
        return self.proj(x)

def build_model(c):
    n=c['name']; ci=c.get('in_channels',5); co=c.get('out_channels',1)
    if n=='unet': return UNet(ci,co,c['base_width'])
    if n=='radiounet': return RadioUNet(ci,co,c['base_width'])
    kw=dict(cin=ci,cout=co,width=c['width'],m1=c['modes1'],m2=c['modes2'],depth=c['depth'],padding=c['padding'])
    if n=='fno': return FNO2d(**kw,use_unet=False)
    if n=='ufno': return FNO2d(**kw,use_unet=True)
    raise ValueError(n)
