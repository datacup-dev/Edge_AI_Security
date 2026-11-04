"""Load and use the completed validation-selected overall native-domain pipeline."""
import json
from pathlib import Path
import numpy as np
import torch
import joblib
from scipy.special import expit
from adapted_models import AdaptedModel
from expanded_models import ExpandedModel
from exact_classical_models import NativeLinear,ExactPolynomialSVM,ExactBlockKNN
from native_gaussian_nb import NativeGaussianNB
from classical_data import sparse_batch

PROJECT=Path(__file__).resolve().parent
NAMES=['DataSense','Edge-IIoTset','CICIoT2023','X-IIoTID']


class OverallBaseline:
    def __init__(self,device=None):
        path=PROJECT/'artifacts/overall_baseline.json'
        if not path.exists():raise RuntimeError('The overall comparison is not complete')
        self.record=json.loads(path.read_text())
        if self.record['status']!='verified_complete':raise RuntimeError('Overall baseline artifact audit is not complete')
        self.selected=self.record['selected'];self.kind=self.selected['model']
        features=[json.loads((PROJECT/'artifacts/encoded_inputs'/n/'preprocessing.json').read_text())['features'] for n in NAMES]
        widths=list(map(len,features));self.offsets=np.cumsum([0]+widths).tolist();self.dimension=self.offsets[-1]+4
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        checkpoint=PROJECT/self.selected['checkpoint'];self.path=checkpoint
        if self.kind in ['MLP','Autoencoder','GNN','Transformer','GAN','LR','SVM']:
            if self.kind=='MLP':self.model=AdaptedModel('MLP',features)
            elif self.kind=='LR':self.model=NativeLinear(widths)
            elif self.kind=='SVM':
                cfg=json.loads((checkpoint.parent/'configuration.json').read_text())['kernel']
                self.model=ExactPolynomialSVM(widths,gamma=cfg['gamma'],coef0=cfg['coef0'])
            else:
                torch.backends.mha.set_fastpath_enabled(False)
                self.model=ExpandedModel(self.kind,features)
            state=torch.load(checkpoint,map_location=self.device,weights_only=True)
            self.model.load_state_dict(state['state_dict']);self.model.to(self.device).eval()
        elif self.kind in ['DT','RF']:self.model=joblib.load(checkpoint)
        elif self.kind=='NB':self.model=NativeGaussianNB.load(checkpoint)
        elif self.kind=='KNN':self.model=ExactBlockKNN.load(checkpoint.parent)
        elif self.kind=='GB':
            import xgboost as xgb
            self.model=xgb.Booster();self.model.load_model(checkpoint)
            self.model.set_param({'device':'cpu','nthread':4})
        else:raise ValueError(self.kind)
        self.features=features

    @torch.inference_mode()
    def predict_probability(self,preprocessed_rows,domain):
        code=NAMES.index(domain);x=np.asarray(preprocessed_rows,dtype=np.float32)
        if x.ndim!=2 or x.shape[1]!=len(self.features[code]):raise ValueError('Expected native features in the saved preprocessing order')
        if not np.isfinite(x).all():raise ValueError('Inputs must be finite and already preprocessed')
        if self.kind in ['MLP','Autoencoder','GNN','Transformer','GAN','LR','SVM']:
            values=torch.from_numpy(x.copy()).to(self.device)
            use_amp=self.device.type=='cuda' and self.kind not in ['LR','SVM']
            with torch.autocast(device_type=self.device.type,dtype=torch.float16,enabled=use_amp):score=self.model(values,code)
            return torch.sigmoid(score.float()).cpu().numpy()
        if self.kind in ['DT','RF']:return self.model.predict_proba(sparse_batch(x,code,self.offsets,self.dimension))[:,1]
        if self.kind=='NB':return expit(self.model.decision_function(x,code))
        if self.kind=='KNN':return self.model.predict_probability(x,code)
        import xgboost as xgb
        matrix=xgb.DMatrix(sparse_batch(x,code,self.offsets,self.dimension),nthread=4)
        return self.model.predict(matrix)


def load_overall_baseline(device=None):return OverallBaseline(device)
