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




class PromptEncoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        
        
        self.fc = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.SiLU(),
            nn.Linear(4096, 4096),
        )
        
    def forward(self, x):
        
        x=self.fc(x)
        
        
        return x
    
if __name__=='__main__':
    pass