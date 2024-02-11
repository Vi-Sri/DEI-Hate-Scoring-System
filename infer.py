import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Define the BertClassifier model
class BertClassifier(nn.Module):
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
        if labels is not None:
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(cls_output, labels)
            return loss, cls_output
        return cls_output

# Initialize the model
bert_model_name = 'bert-base-cased'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertClassifier(BertModel.from_pretrained(bert_model_name), 6)

model.load_state_dict(torch.load('model.pt'))
model.to(device)

model.eval()  # Set the model to evaluation mode

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

# Function to prepare input text
def prepare_input(text, tokenizer, max_length=120):
    tokens = tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_length, 
                                   truncation=True, return_tensors="pt", 
                                   pad_to_max_length=True, return_attention_mask=True)
    return tokens['input_ids'].to(device), tokens['attention_mask'].to(device)

# Inference function
def predict(text, model, tokenizer):
    model.eval()  # Ensure model is in evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        input_ids, attention_mask = prepare_input(text, tokenizer)
        logits = model(input_ids, attention_mask=attention_mask)
        probabilities = torch.sigmoid(logits)  # Convert logits to probabilities
        return probabilities.squeeze()

# Example text for inference
text = "You are the worst person I have ever met!"
probabilities = predict(text, model, tokenizer)
print(probabilities.cpu().numpy()) 