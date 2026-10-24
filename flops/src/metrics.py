import numpy as np

def batch_sums(p,y,min_db,max_db):
    p=np.clip(p,0,1); scale=max_db-min_db
    pd=p*scale+min_db; yd=y*scale+min_db; e=pd-yd
    return {'se':float((e*e).sum()),'ae':float(np.abs(e).sum()),'n':e.size,
            'nse':float(((p-y)**2).sum()),'den':float((y*y).sum())}

def finalize(s):
    return {'rmse_db':(s['se']/s['n'])**.5,'mae_db':s['ae']/s['n'],
            'nrmse':(s['nse']/s['n'])**.5,'relative_l2':(s['nse']/max(s['den'],1e-12))**.5}
