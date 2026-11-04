"""Both-class training-only attack run: frozen native-schema GB oracle."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import xgboost as xgb
from adapted_models import AdaptedModel,group_for

PROJECT=Path(__file__).resolve().parent
ROOT=PROJECT/'artifacts/mixed_training_attacks'
NAMES=['DataSense','Edge-IIoTset','CICIoT2023','X-IIoTID']
EPS=.3
COUNT=35030746


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def project_path(relative):
    """Resolve project-relative metadata saved on either Windows or POSIX."""
    return PROJECT/str(relative).replace(chr(92),'/')


def replace(source,target):
    for attempt in range(100):
        try:Path(source).replace(target);return
        except PermissionError:
            if attempt==99:raise
            time.sleep(.05)


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');replace(temporary,path)


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):result.update(block)
    return result.hexdigest()


class Rule:
    def __init__(self,features,low,high):
        self.low=np.array(low,np.float32);self.high=np.array(high,np.float32)
        self.mask=np.array([group_for(n) in (1,2,3) and not any(s in n.lower() for s in ['port','protocol','ips_','macs_','header','flag','checksum']) for n in features])&(self.high>self.low)
    def bounds(self,x):
        exact=np.asarray(x,np.float64);low=np.maximum(exact-EPS,np.minimum(self.low,exact));high=np.minimum(exact+EPS,np.maximum(self.high,exact))
        lo=low.astype(np.float32);hi=high.astype(np.float32)
        return np.where(lo<low,np.nextafter(lo,np.float32(np.inf)),lo),np.where(hi>high,np.nextafter(hi,np.float32(-np.inf)),hi)
    def project(self,z,x):
        lo,hi=self.bounds(x);return np.where(self.mask,np.clip(z,lo,hi),x).astype(np.float32)
    def tensor(self,z,x):
        lo=torch.maximum(x-EPS,torch.minimum(torch.as_tensor(self.low,device=x.device),x));hi=torch.minimum(x+EPS,torch.maximum(torch.as_tensor(self.high,device=x.device),x))
        return torch.where(torch.as_tensor(self.mask,device=x.device),torch.minimum(torch.maximum(z,lo),hi),x)


class Data:
    def __init__(self):
        self.mixed_path=PROJECT/'artifacts/mixed_training/records.npy';self.records=np.load(self.mixed_path,mmap_mode='r')
        assert len(self.records)==COUNT
        self.info=[read(PROJECT/'artifacts/encoded_inputs'/n/'preprocessing.json') for n in NAMES]
        self.features=[i['features'] for i in self.info];self.widths=list(map(len,self.features));self.offsets=np.cumsum([0]+self.widths).tolist()
        self.x=[np.load(PROJECT/'artifacts/encoded_inputs'/n/'features.npy',mmap_mode='r') for n in NAMES]
        ranges=read(PROJECT/'artifacts/classical_native/manifest.json')
        self.rules=[Rule(f,ranges['training_min'][c],ranges['training_max'][c]) for c,f in enumerate(self.features)]
    def local(self,records,code):return np.asarray(self.x[code][records['source_row']],np.float32)
    def compact(self,records):
        values=np.full((len(records),64),np.nan,np.float32)
        for code in range(4):
            selected=np.flatnonzero(records['dataset_id']==code)
            values[selected,:self.widths[code]]=self.local(records[selected],code)
        return values


class Target:
    def __init__(self,data,threads=2):
        self.data=data;baseline=read(PROJECT/'artifacts/overall_baseline.json')
        assert baseline['status']=='verified_complete' and baseline['selected']['model']=='GB'
        self.path=project_path(baseline['selected']['checkpoint']);self.hash=digest(self.path)
        self.model=xgb.Booster();self.model.load_model(self.path);self.model.set_param({'device':'cpu','nthread':threads});self.queries=0
    def local(self,x,code):
        matrix=np.full((len(x),157),np.nan,np.float32);a,b=self.data.offsets[code:code+2]
        matrix[:,a:b]=x;matrix[:,153+code]=1;self.queries+=len(x)
        return self.model.inplace_predict(matrix).astype(np.float32)
    def compact(self,values,records):
        p=np.empty(len(records),np.float32)
        for code in range(4):
            selected=np.flatnonzero(records['dataset_id']==code)
            if len(selected):p[selected]=self.local(values[selected,:self.data.widths[code]],code)
        return p


def proxy(data,distilled=True):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=AdaptedModel('MLP',data.features).to(device)
    path=ROOT/'surrogate/model.pt' if distilled else PROJECT/'artifacts/adapted_experiment/MLP/best.pt'
    model.load_state_dict(torch.load(path,map_location=device,weights_only=True)['state_dict']);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model,device


def load_shard(method,path):
    """Reconstruct clean/candidate native rows and provenance without duplication."""
    data=Data()
    with np.load(path) as z:
        assert str(z['attack'])==method and str(z['source_split'])=='train'
        positions=z['mixed_position'];records=data.records[positions]
        return dict(records=np.asarray(records).copy(),mixed_position=positions.copy(),x_clean_native=data.compact(records),
            x_adversarial_native=z['x_adversarial_native'].copy(),labels=z['labels'].copy(),target_labels=z['target_labels'].copy(),success=z['success'].copy())
