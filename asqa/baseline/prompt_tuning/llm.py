
from transformers import AutoTokenizer,TextStreamer,TextIteratorStreamer,AutoModelForCausalLM
import torch
import yaml


from dataset import collateLM

with open('config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open('config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)
class LLaMa:
    def __init__(self, model_dir):
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, lstrip=False, token='access_tokens')
        self.tokenizer.model_max_length=2048
        self.eos_id=self.tokenizer.eos_token_id
        self.eos=self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

        self.generate_config=config['generate_config']
        
        self.model=AutoModelForCausalLM.from_pretrained(model_dir, token='access_tokens', device_map='cpu', torch_dtype=torch.float16)
        self.model=self.model.bfloat16()
        self.model.to(device)
        self.model.training=True
        self.model.requires_grad_(False)
        self.chat_history = []
        self.system_prompt = ' '
    
    
    def forward(self, ids, target=None, masks=None, soft_examples=None):
        
        batchsize=soft_examples.shape[0]
        bos=torch.ones([batchsize, 1],dtype=ids.dtype).to(device) * self.tokenizer.bos_token_id
        bos_embeds=self.model.model.embed_tokens(bos.to(device))
        prompts_embeds=self.model.model.embed_tokens(ids) 

        input_embeds=torch.cat([bos_embeds,soft_examples,prompts_embeds],dim=1)

        atts_exam=torch.ones(soft_examples.size()[:-1]).to(soft_examples.device)
        attns_bos=atts_exam[:,:1]
        attns=torch.cat([attns_bos,atts_exam,masks],dim=1)
        
        target=torch.cat([torch.ones((soft_examples.shape[0],soft_examples.shape[1]+1),device=device)*-100,target],dim=1)
        
        target=target.masked_fill(
            target==self.tokenizer.pad_token_id,-100
        )

        input_embeds=input_embeds.to(torch.bfloat16)
        attns=attns.to(torch.bfloat16)
        target=target.to(torch.long)
        
        output = self.model(inputs_embeds=input_embeds,
                            attention_mask=attns,
                            labels=target,
                            use_cache=True)

        return output.logits, output.loss,target

    @torch.inference_mode()
    def generate(self, message, soft_examples=None, max_new_tokens = 1024, streamer=True,return_eigenscore=False):
        
        if streamer:
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        else:
            streamer=None
        if soft_examples==None:
            tokens = self.tokenizer(message, return_tensors='pt').input_ids

            generate_ids = self.model.generate(input_ids=tokens.cuda(), streamer=streamer,max_new_tokens=max_new_tokens,**self.generate_config)

            output = self.tokenizer.decode(generate_ids[0,tokens.shape[1]:],skip_special_tokens=True).strip()
            output=output.split('\n')[0]
            return output
        
        tokens = self.tokenizer(message, return_tensors='pt').to(device)
        ids=tokens.input_ids[:,1:]
        batchsize=soft_examples.shape[0]
        bos=torch.ones([batchsize, 1],dtype=ids.dtype).to(device) * self.tokenizer.bos_token_id
        bos_embeds=self.model.model.embed_tokens(bos.to(device))
        prompts_embeds=self.model.model.embed_tokens(ids)
        
        input_embeds=torch.cat([bos_embeds,soft_examples,prompts_embeds],dim=1)

        atts_exam=torch.ones(soft_examples.size()[:-1],).to(soft_examples.device)
        attns_bos=atts_exam[:,:1]
        attns=torch.cat([attns_bos,atts_exam,tokens.attention_mask[:,1:]],dim=1)
        
        input_embeds=input_embeds.to(torch.bfloat16)
        attns=attns.to(torch.bfloat16)
        
        generate_ids = self.model.generate(inputs_embeds=input_embeds,
                        attention_mask=attns,
                        max_new_tokens=max_new_tokens,
                        streamer=streamer,
                        **self.generate_config)
        # print(generate_ids)
        output = self.tokenizer.decode(generate_ids[0],skip_special_tokens=True).strip()
        if return_eigenscore==True:
            hh=[]
            k=10
            for i in range(k):
                dic = self.model.generate(inputs_embeds=input_embeds,
                        attention_mask=attns,
                        max_new_tokens=max_new_tokens,
                        streamer=streamer,
                        return_dict_in_generate=True,
                        output_hidden_states=True,
                        **config['generate_config_es'])
                
                h=dic.hidden_states[-1][16][:,-1,:].squeeze()
                hh.append(h.cpu().float())
            hh=torch.stack(hh)
            cov=hh@(torch.eye(4096)-1/4096*torch.ones((1,k))@torch.ones((k,1)))@hh.T
            eigenscore=1/k*torch.logdet(cov+0.001*torch.eye(k))
            return output,eigenscore.item()
        return output


if __name__=="__main__":
    pass
    