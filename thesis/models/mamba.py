import torch 
from transformers import MambaConfig, MambaModel
from torch.nn import Linear, CrossEntropyLoss



class MambaClassifier(torch.nn.Module):
    def __init__(self,input_size,num_hidden_layers=5):
        super().__init__()
        configuration = MambaConfig()
        configuration.hidden_size = input_size
        configuration.num_hidden_layers = num_hidden_layers
        print(configuration)
        self.base = MambaModel(configuration)
        self.classifier = Linear(configuration.hidden_size, 1)

    def forward(self,inputs):
        outputs = self.base(inputs_embeds=inputs).last_hidden_state
        logits = self.classifier(outputs[:, 0])  # CLS token representation
        return logits


if __name__ == "__main__":
    input_sample = torch.randn((100,42,2000)) #batch,num_layers,dim
    model = MambaClassifier()
    output = model(input_sample)
    
    print(output)