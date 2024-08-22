import torch
import torch.nn as nn
from transformers import AutoModel ,AutoTokenizer
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
from model import SBert
from dataset import QADataset,collect_fn
import yaml
from torch.utils.data import random_split

with open('config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)

seed = config['seed']

generator = torch.Generator()
generator.manual_seed(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def cos_sim(a:torch.Tensor, b:torch.Tensor):
    return (a @ b.T)/(torch.norm(a,dim=1)[:,None]@torch.norm(b,dim=1)[None,:])

class Retriever(torch.nn.Module):
    def __init__(self,k=config['train_config']['topk']):
        super().__init__()
        self.model=SBert()
        self.model=self.model.to(device)
        self.model.eval()
        self.k=k
        try:
            self.saves=torch.load('save/embs.pt')
            self.embeddings=self.saves['embs'].to(device)
            self.examples=self.saves['examples']
        except:
            print('start to get embeddings')
            self.get_embeddings()
            self.saves=torch.load('save/embs.pt')
            self.embeddings=self.saves['embs'].to(device)
            self.examples=self.saves['examples']
    @torch.inference_mode()
    def get_embeddings(self):
       
        embeddings=[]
        examples_ls=[]
        dataset=QADataset()
        train_size = int(config['train_config']['spilt'] * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, _ = random_split(dataset, [train_size, test_size],generator=generator)
        dataloader = DataLoader(train_dataset, batch_size=256, shuffle=False,collate_fn=collect_fn)
        for x , q_a in (bar:=tqdm(dataloader,ncols=0)):
            x=x.to(device)
            embedding  = self.model(x)#(bs, d=384)
            embeddings.append(embedding)
            examples_ls.extend(q_a)
        embeddings=torch.cat(embeddings)#10000*384
        save_dict={'embs':embeddings,
                   'examples':examples_ls,
                   }
        torch.save(save_dict,'save/embs.pt')

    @torch.inference_mode()
    def retrieve(self, query,train=True,random_select=False,largest=True):
        
        if type(query[0])==str:
            query = self.model.tokenizer(text=query,return_tensors="pt",padding=True,truncation=True,max_length=128)
            query = query.to(device)
        query_embedding = self.model(query)
        if len(query_embedding.shape)==1:
            query_embedding=query_embedding[None,:]
        #cosine similarity
        sim = cos_sim(query_embedding,self.embeddings)
        
        #top-k vector and index
        if train:
            v, id = torch.topk(sim, k=self.k+1, dim=1, largest=largest)
            id=id.cpu().numpy()
            
            topk_examples_per_batch = []
            for batch_index in range(0,id.shape[0]):
                topk_examples = [self.examples[i] for i in id[batch_index][1:]]
                topk_examples='\n'.join(topk_examples)
                topk_examples_per_batch.append(topk_examples)
            return topk_examples_per_batch
        else:
            v, id = torch.topk(sim, k=self.k, dim=1, largest=largest)
            id=id.cpu().numpy()
            if random_select:
                random_integers = torch.randperm(len(self.embeddings))[:id.shape[0]*id.shape[1]]
                id = random_integers.view(id.shape[0],id.shape[1])
            topk_examples_per_batch = []
            for batch_index in range(id.shape[0]):
                topk_examples = [self.examples[i] for i in id[batch_index]]
                topk_examples='\n'.join(topk_examples)
                topk_examples_per_batch.append(topk_examples)
            return topk_examples_per_batch
            
    
if __name__=='__main__':
    pass