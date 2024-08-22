
from transformers import AutoTokenizer,TextStreamer,TextIteratorStreamer,AutoModelForCausalLM
import torch
import yaml



with open('config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open('config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)
class LLaMa:
    def __init__(self, model_dir,hf_access_token=config['hf_access_tokens']):
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, lstrip=False, token=hf_access_token)
        self.tokenizer.model_max_length=2048
        self.eos_id=self.tokenizer.eos_token_id
        self.eos=self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        
        self.generate_config=config['generate_config']
        
        self.model=AutoModelForCausalLM.from_pretrained(model_dir, token=hf_access_token, device_map='cpu', torch_dtype=torch.float16)
        self.model=self.model.bfloat16()
        self.model.to(device)
        self.model.training=True
        self.model.requires_grad_(False)
        self.chat_history = []
        self.system_prompt = "You are a QA system."
        
       
    def forward(self, ids, target=None, masks=None, soft_examples=None,output_hidden_states=False):
        
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
                            use_cache=True,
                            output_hidden_states=output_hidden_states)
        if output_hidden_states: 
            hh=soft_examples.shape[1]+1
            h=output.hidden_states[16][:,-1,:]

           
            return output.logits, output.loss,target,h
        return output.logits, output.loss,target

    @torch.inference_mode()
    def generate(self, message, soft_examples=None, max_new_tokens = 1024, streamer=True,return_eigenscore=False,return_hidden_states=False,return_cov=False,k=10):
        
        if streamer:
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        else:
            streamer=None
        if soft_examples==None:
            tokens = self.tokenizer(message, return_tensors='pt').input_ids

            generate_ids = self.model.generate(input_ids=tokens.cuda(), max_new_tokens =max_new_tokens ,streamer=streamer,**self.generate_config)

            output = self.tokenizer.decode(generate_ids[0, len(tokens[0]):-1],skip_special_tokens=True)
            if return_eigenscore==True:
                hh=[]
                k=k
                predicts=[]
                for i in range(k):
                    dic = self.model.generate(input_ids=tokens.cuda(),
                            max_new_tokens=max_new_tokens,
                            streamer=streamer,
                            return_dict_in_generate=True,
                            output_hidden_states=True,
                            **config['generate_config_es'])
                    
                    h=dic.hidden_states[-1][16][:,-1,:].squeeze()
                    hh.append(h.cpu().float())
                    predicts.append(self.tokenizer.decode(dic.sequences[0,len(tokens[0]):-1],skip_special_tokens=True))

                hh=torch.stack(hh)
                cov=hh@(torch.eye(4096)-1/4096*torch.ones((1,k))@torch.ones((k,1)))@hh.T
                eigenscore=1/k*torch.logdet(cov+0.001*torch.eye(k))
                if return_cov==True:
                    return output,eigenscore.item(),cov+0.001*torch.eye(k),predicts
                return output,eigenscore.item()
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

        output = self.tokenizer.decode(generate_ids[0],skip_special_tokens=True)
        if return_eigenscore==True:
            hh=[]
            k=k
            predicts=[]
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
                predicts.append(self.tokenizer.decode(dic.sequences[0],skip_special_tokens=True))
            hh=torch.stack(hh)
            cov=hh@(torch.eye(4096)-1/4096*torch.ones((1,k))@torch.ones((k,1)))@hh.T
            eigenscore=1/k*torch.logdet(cov+0.001*torch.eye(k))
            if return_cov==True:
                    return output,eigenscore.item(),cov+0.001*torch.eye(k),predicts
            return output,eigenscore.item()
        if return_hidden_states==True:
            dic = self.model.generate(inputs_embeds=input_embeds,
                    attention_mask=attns,
                    max_new_tokens=max_new_tokens,
                    streamer=streamer,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    **self.generate_config)
            h=dic.hidden_states[-1][16][:,-1,:].squeeze()
            return output,h
        
        return output


if __name__=="__main__":
    pass
