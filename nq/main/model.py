import torch
import torch.nn as nn
from transformers import AutoModel ,AutoTokenizer
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from dataset import QADataset,collect_fn
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)

class SBert(torch.nn.Module):
    def __init__(self):
        super(SBert, self).__init__()
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')

    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, x):
        if type(x[0])==str:
            x=self.tokenizer(text=x,return_tensors="pt",padding=True,truncation=True,max_length=512)
        out = self.model(**x)
        sentence_embeddings = self.mean_pooling(out, x['attention_mask'])
        return sentence_embeddings

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x) :
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class PromptEncoder(nn.Module):
    def __init__(self, input_size=4096, output_size=4096,bottle=config['train_config']['pe_bottle'],heads=config['train_config']['pe_heads']):
        super().__init__()
        
        # self.q=nn.Parameter(data=torch.randn(1, 32, 4096), requires_grad=True)
        # self.multihead_attn = nn.MultiheadAttention(embed_dim=input_size, num_heads=heads,batch_first=True,dropout=config['train_config']['pe_dropout'])
        # self.pos_encoder = PositionalEncoding(d_model=input_size, dropout=config['train_config']['pe_dropout'])
        # self.layer_norm = nn.LayerNorm(input_size)
        self.fc = nn.Sequential(
            nn.Linear(4096, bottle),
            nn.Dropout1d(config['train_config']['pe_dropout']),
            nn.SiLU(),
        )
        self.mean= nn.Sequential(
            nn.Linear(bottle, output_size),
        )
        # self.variance= nn.Sequential(
        #     nn.Linear(bottle, output_size),
        # )
        # with torch.no_grad():
        #     for param in self.variance.parameters():
        #         nn.init.constant_(param, 0)
        
    def forward(self, x,attns=None):
        # q = self.q.expand(x.shape[0], -1, -1)
        # x=self.pos_encoder(x)
        # x,_=self.multihead_attn(q,x,x,key_padding_mask=attns)
        # q=self.layer_norm(q+x)
        
        # x=self.transformer_encoder(x,src_key_padding_mask=attns)
        x=self.fc(x)
        mean=self.mean(x)
        
        logvar=None
        
        return mean, logvar
    
    def reparameter(self, mean,logvar):
        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std)
        z=mean+eps*std
        return z





if __name__=='__main__':
    model=PromptEncoder()
    x=torch.randn((8,105,4096))
    y,_=model(x)
    print(y.shape)
    # dataset=QADataset()
    # train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True,collate_fn=collect_fn)
    # model=SBert()
    # model.to(device)
    # for x in (bar:=tqdm(train_dataloader,ncols=0)):
    #     x=x.to(device)

    #     y  = model(x)
    #     print(y.shape)
    #     exit()