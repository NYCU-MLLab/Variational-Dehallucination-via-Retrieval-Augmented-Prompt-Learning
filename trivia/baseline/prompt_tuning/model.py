import torch
import torch.nn as nn
from transformers import AutoModel ,AutoTokenizer
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from dataset import QADataset,collect_fn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)




class Softprompt(nn.Module):
    def __init__(self, ):
        super().__init__()
        
        self.soft_emb=nn.parameter.Parameter(data=torch.randn((1,20,4096)), requires_grad=True)
        
    def forward(self,x ):
        soft_emb = self.soft_emb.expand(x.shape[0], -1, -1)
        
        
        
        return soft_emb
    
    
if __name__=='__main__':
    pass