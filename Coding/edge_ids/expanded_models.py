"""Additional native-domain classifiers; no temporal/topology equivalence claims."""
import torch
from torch import nn
from torch.nn import functional as F
from adapted_models import FeatureEncoder


GRAPH_EDGES = [(0,5),(1,2),(1,3),(1,4),(1,6),(2,3),(2,6),
               (3,5),(4,5),(4,6),(5,6),(5,7)]


class ExpandedModel(nn.Module):
    def __init__(self,kind,feature_lists):
        super().__init__()
        self.kind=kind
        self.encoders=nn.ModuleList(FeatureEncoder(f) for f in feature_lists)
        if kind=='Autoencoder':
            self.core=nn.Sequential(nn.Flatten(),nn.Linear(64,32),nn.ReLU(),
                                    nn.Dropout(.3),nn.Linear(32,16),nn.ReLU())
            self.decoders=nn.ModuleList(nn.Sequential(nn.Linear(16,64),nn.ReLU(),nn.Linear(64,len(f))) for f in feature_lists)
            self.head=nn.Sequential(nn.Linear(16,32),nn.ReLU(),nn.Dropout(.3),nn.Linear(32,1))
        elif kind=='GNN':
            adjacency=torch.zeros(8,8)
            for a,b in GRAPH_EDGES:
                adjacency[a,b]=adjacency[b,a]=1
            adjacency/=adjacency.sum(1,keepdim=True).clamp_min(1)
            self.register_buffer('adjacency',adjacency)
            self.self_layers=nn.ModuleList([nn.Linear(8,32),nn.Linear(32,32),nn.Linear(32,16)])
            self.neighbor_layers=nn.ModuleList([nn.Linear(8,32,bias=False),nn.Linear(32,32,bias=False),nn.Linear(32,16,bias=False)])
            self.head=nn.Sequential(nn.Flatten(),nn.Linear(128,64),nn.ReLU(),nn.Dropout(.3),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
        elif kind=='Transformer':
            self.project=nn.Linear(8,64)
            self.positions=nn.Parameter(torch.randn(1,8,64)*.02)
            layer=nn.TransformerEncoderLayer(64,4,128,dropout=.3,activation='gelu',batch_first=True,norm_first=True)
            self.core=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
            self.head=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(.3),nn.Linear(32,1))
        elif kind=='GAN':
            self.core=nn.Sequential(nn.Flatten(),nn.Linear(64,128),nn.ReLU(),nn.Dropout(.3),nn.Linear(128,64),nn.ReLU(),nn.Dropout(.3))
            self.head=nn.Linear(64,3)
            self.generators=nn.ModuleList(nn.Sequential(nn.Linear(34,64),nn.ReLU(),nn.Linear(64,128),nn.ReLU(),nn.Linear(128,len(f))) for f in feature_lists)
        else:
            raise ValueError(kind)

    def latent(self,x,domain):
        tokens=self.encoders[domain](x)
        if self.kind in {'Autoencoder','GAN'}:
            return self.core(tokens)
        if self.kind=='Transformer':
            return self.core(self.project(tokens)+self.positions).mean(1)
        h=tokens
        for own,neighbors in zip(self.self_layers,self.neighbor_layers):
            neighbor_mean=torch.einsum('ij,bjf->bif',self.adjacency,h)
            h=F.dropout(F.relu(own(h)+neighbors(neighbor_mean)),.3,self.training)
        return h

    def discriminate(self,x,domain):
        return self.head(self.latent(x,domain))

    def generate(self,labels,domain):
        noise=torch.randn(len(labels),32,device=labels.device)
        conditional=F.one_hot(labels.long(),2).float()
        return self.generators[domain](torch.cat([noise,conditional],dim=1))

    def forward(self,x,domain):
        logits=self.discriminate(x,domain)
        return logits[:,1]-logits[:,0] if self.kind=='GAN' else logits.squeeze(-1)

    def supervised_loss(self,x,y,domain):
        if self.kind=='Autoencoder':
            z=self.latent(x,domain)
            classification=F.binary_cross_entropy_with_logits(self.head(z).squeeze(-1),y)
            reconstruction=F.mse_loss(self.decoders[domain](z),x)
            return classification+.1*reconstruction,classification.detach(),reconstruction.detach()
        classification=F.binary_cross_entropy_with_logits(self(x,domain),y)
        return classification,classification.detach(),classification.detach()*0
