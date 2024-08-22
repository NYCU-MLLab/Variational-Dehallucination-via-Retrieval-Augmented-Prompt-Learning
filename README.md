# Variational-Dehallucination-via-Retrieval-Augmented-Prompt-Learning

## Environment install
```
  conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
  
  conda install -y conda-forge::tqdm
  
  conda install -y conda-forge::datasets
  
  conda install -y anaconda::transformers
  
  pip install sentence-transformers
  
  conda install -y fastai::accelerate
  
  conda install -y conda-forge::bert_score
  
  conda install -y conda-forge::rouge-score
  
```  

## Run
- Natural Questions
  - main
  ```
    cd nq/main
    bash run.sh
  ```
  - baseline
    - original model and Kate and random selection
    ```
      cd nq/baseline/kate_orig_rs
      bash run.sh
    ```
    - KP
    ```
      cd nq/baseline/kate_pt
      bash run.sh
    ```
    - p_tuning
    ```
      cd nq/baseline/p_tuning
      bash run.sh
    ```
    - prompt tuning
    ```
      cd nq/baseline/prompt_tuning
      bash run.sh
    ```

- TriviaQA
  - main
  ```
    cd trivia/main
    bash run.sh
  ```
  - baseline
    - original model and Kate and random selection
    ```
      cd trivia/baseline/kate_orig_rs
      bash run.sh
    ```
    - KP
    ```
      cd trivia/baseline/kate_pt
      bash run.sh
    ```
    - p_tuning
    ```
      cd trivia/baseline/p_tuning
      bash run.sh
    ```
    - prompt tuning
    ```
      cd trivia/baseline/prompt_tuning
      bash run.sh
    ```

- ASQA
  - main
  ```
    cd asqa/main
    bash run.sh
  ```
  - baseline
    - original model and Kate and random selection
    ```
      cd asqa/baseline/kate_orig_rs
      bash run.sh
    ```
    - KP
    ```
      cd asqa/baseline/kate_pt
      bash run.sh
    ```
    - p_tuning
    ```
      cd asqa/baseline/p_tuning
      bash run.sh
    ```
    - prompt tuning
    ```
      cd asqa/baseline/prompt_tuning
      bash run.sh
    ```
