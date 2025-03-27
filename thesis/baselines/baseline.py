
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import spacy
from thesis.baselines.selfcheckgpt.selfcheckgpt.modeling_selfcheck import SelfCheckLLMPrompt
from thesis.data_handling.benchmark import get_df 
from thesis.metrics import calculate_metrics, calculate_score_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Baseline():
#parent class for all baselines 
#defines abstract test method 
    def __init__(self):
        pass
    
    def detect():
        raise NotImplementedError("Subclasses should implement this!")
    
    def require_thresholding():
        raise NotImplementedError("Subclasses should implement this!")
    

class CoVe(Baseline):
    pass

class SelfConsistency(Baseline):
    print("no detection method :(")
    pass

class SelfCheckGPT(Baseline):
    num_samples = 20 #used in paper 
    
    def require_thresholding(self):
        return True
    
    def get_samples(self,model_id,question):
        tokenizer = AutoTokenizer.from_pretrained(model_id,device_map="auto")
        model = AutoModelForCausalLM.from_pretrained(model_id,device_map="auto")
        responses = []
        inputs = tokenizer(question, padding=True, truncation=True, return_tensors="pt").to('cuda')
        input_ids = inputs['input_ids'] 
        with torch.no_grad():
            output = model.generate(
            input_ids,
            do_sample=True,  # Enable sampling for randomness
            num_return_sequences= self.num_samples,  # Generate multiple outputs
        )
        for sequence in output:
            response = tokenizer.decode(sequence, skip_special_tokens=True)
            responses.append(response)
        return responses
            
    def detect(self,model_id,question,answer): 
        samples = self.get_samples(model_id,question)
        selfcheck_prompt = SelfCheckLLMPrompt(model_id, device)
        nlp = spacy.load("en_core_web_sm")
        sentences = [
            sent.text.strip() for sent in nlp(answer).sents
        ]  # spacy sentence tokenization
        print(sentences)

        sent_scores_prompt = selfcheck_prompt.predict(
            sentences=sentences,  # list of sentences
            sampled_passages=samples,  # list of sampled passages
            verbose=True,  # whether to show a progress bar
        )

        print(sent_scores_prompt)
        hallucination_score = np.mean(sent_scores_prompt)
        print("Hallucination Score:", hallucination_score)
    
        return hallucination_score

class DirectPrompt(Baseline):
    pass

class AvgProb(Baseline):
    #entropy/maxprob 
    pass

class Perplexity(Baseline):
    #entropy/maxprob 
    pass


class PTrue(Baseline):
    pass

def get_baseline(cfg):
    
    if cfg.task.baseline_name == "selfcheckgpt":
        baseline = SelfCheckGPT() 
    return baseline

def evaluate_baseline(cfg):
    
    #load df 
    df = get_df(cfg)[:10]

    #load baseline
    baseline = get_baseline(cfg)
    
    #iterate over rows 
    results = []
    for index, row in df.iterrows():
        question = row["question"]
        answer = row["answer"]
        fake_answer = row["fake_answer"]
        #perform detection
        result = baseline.detect(cfg.llm.name,question,answer)
        results.append(result)
        
        #perform detection
        result = baseline.detect(cfg.llm.name,question,fake_answer)
        results.append(result)
    
    labels = [0,1] * len(df)
    
    results = np.array(results)
    labels = np.array(labels)
    
    #calculate metrics 
    if baseline.require_thresholding:
        #calculate auc 
        roc_auc, average_precision = calculate_score_metrics(scores = results, labels=labels)
        print(f"ROC AUC: {roc_auc}")
        print(f"AUPRC: {average_precision}")
        results = results > 0.5 
    
    #calculate accuracy, precision, recall, f1
    accuracy, precision, recall, f1 = calculate_metrics(preds=results,labels=labels)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

if __name__ == "__main__":
    #perform small check
    model_id = "google/gemma-2-9b-it"
    question = "What is the capital of France?"
    answer = "The capital of France is Paris."
    scg = SelfCheckGPT()
    scg.detect(model_id,question,answer)
    