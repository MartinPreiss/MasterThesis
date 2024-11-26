import torch 
from transformers import MambaConfig, MambaModel
from torch.nn import Linear, CrossEntropyLoss



class MambaClassifier(torch.nn.Module):
    def __init__(self,input_size,num_hidden_layers=5):
        super().__init__()
        configuration = MambaConfig()
        #configuration.hidden_size = input_size
        configuration.vocab_size = 1
        configuration.intermediate_size = 500
        configuration.num_hidden_layers = num_hidden_layers
        #print(configuration)
        
        self.down_projection = Linear(input_size,configuration.hidden_size,bias=False)

        self.base = MambaModel(configuration)
        self.classifier = Linear(configuration.hidden_size, 1,bias=False)

    def forward(self,inputs,return_encoded_space=False):
        inputs = self.down_projection(inputs)
        outputs = self.base(inputs_embeds=inputs).last_hidden_state[:,-1]
        logits = self.classifier(outputs)  # CLS token representation
        if return_encoded_space:
            return logits,outputs
        return logits, None


if __name__ == "__main__":
    
    input_size = 2000
    input_sample = torch.randn((100,42,input_size)) #batch,num_layers,dim
    model = MambaClassifier(input_size=input_size,num_hidden_layers=5)
    output = model(input_sample)
    
    #print("simple classifier model")
    #print(model)
    #print(model.base.config)
    
    #print standard mamba config 
    
    from transformers import AutoConfig
    from thesis.utils import print_number_of_parameters

    print_number_of_parameters(model) # vs 
    """
    7679644
    15393500
    # Specify the model name
    model_name = "state-spaces/mamba-130m"

    # Load the model configuration
    config = AutoConfig.from_pretrained(model_name)

    # Print the configuration
    print(config)

    
    """