import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np

# Define the DEIHateClassifier model
class DEIHateClassifier(nn.Module):
    def __init__(self, bert: BertModel, num_classes: int):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Linear(bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, 
                position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
        cls_output = outputs[1]  # The "pooler_output" is the second output
        cls_output = self.classifier(cls_output)
        return cls_output

# Function to prepare input text
def prepare_input(text, tokenizer, max_length=120):
    tokens = tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_length, 
                                   truncation=True, return_tensors="pt", 
                                   pad_to_max_length=True, return_attention_mask=True)
    return tokens['input_ids'], tokens['attention_mask']

# Initialize the model and tokenizer
bert_model_name = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model = DEIHateClassifier(BertModel.from_pretrained(bert_model_name), 6)
model_path = 'model.pt'  # Update this path
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Function to read text file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = file.readlines()
    return [sentence.strip() for sentence in sentences]

# Function to classify sentences
def classify_sentences(sentences, model, tokenizer, threshold=0.5):
    hateful_sentences = []
    model.eval()
    with torch.no_grad():
        for sentence in sentences:
            input_ids, attention_mask = prepare_input(sentence, tokenizer)
            logits = model(input_ids, attention_mask=attention_mask)
            probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()
            if np.any(probabilities > threshold):
                hateful_sentences.append(sentence)
    return hateful_sentences

# Function to compute DEI score
def compute_dei_score(total_sentences, hateful_sentences):
    non_hateful_sentences = total_sentences - len(hateful_sentences)
    hate_score = (non_hateful_sentences / total_sentences) * 100 
    return 100 - hate_score

# Main process
def process_text_file_and_compute_dei(file_path):
    sentences = read_text_file(file_path)
    hateful_sentences = classify_sentences(sentences, model, tokenizer, threshold=0.5)
    dei_score = compute_dei_score(len(sentences), hateful_sentences)
    
    print(f"\n\nDEI Score: {dei_score}%")
    print("Hateful Sentences Identified:")
    for sentence in hateful_sentences:
        print(f"- {sentence}")

# Example usage
file_path = 'email.txt'  # Update this to the path of your text file
process_text_file_and_compute_dei(file_path)
