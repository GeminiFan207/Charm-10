import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class CharmC10NLP(nn.Module):
    def __init__(self, model_name="meta-llama/Llama-2-13b"):
        super(CharmC10NLP, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.encoder.config.hidden_size, 2048)
        self.activation = nn.GELU()
    
    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids, attention_mask=attention_mask)
        x = self.fc(output.last_hidden_state[:, 0, :])
        return self.activation(x)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b")
model = CharmC10NLP()
