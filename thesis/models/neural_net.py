
import torch.nn as nn
import torch
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, num_layers=3):
        super(MLP, self).__init__()
        
        # List to hold the layers
        self.layers = nn.ModuleList()
        current_size = input_size
        
        # Dynamically create hidden layers
        for i in range(num_layers - 1):
            next_size = current_size // 2  # Halving the size each time
            self.layers.append(nn.Linear(in_features=current_size, out_features=next_size))
            
            current_size = next_size
        
        # Final output layer (reducing to 1 or a predefined output size)
        self.output_layer = nn.Linear(in_features=current_size, out_features=1)

        # Activation function
        self.relu = nn.LeakyReLU()
        
        #weight init
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight,nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.output_layer.weight,nonlinearity="leaky_relu")

    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
        
        x = self.output_layer(x)
        return x

class LRProbe(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, iid=None):
        return self.net(x)
    
class AllLayerClassifier(nn.Module):
    def __init__(self, embedding_size, num_llm_layers):
        super().__init__()
        
        self.layer_classifiers = nn.Conv1d(num_llm_layers,num_llm_layers,embedding_size)
        self.activation = nn.ReLU()
        self.aggregate = nn.Linear(num_llm_layers,1)

    def forward(self, x, return_encoded_space=False):
        
        result =  self.aggregate(self.activation(self.layer_classifiers(x)).squeeze(dim=-1))
        if return_encoded_space:
            raise Exception("Not implemented")
        return result, None



class LayerFusion(nn.Module):
    def __init__(self,num_llm_layers, embedding_size):
        super().__init__()
        self.gating = nn.Conv1d(num_llm_layers,num_llm_layers,embedding_size)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(embedding_size,1)

    def forward(self, x,return_encoded_space=False):
        weights = F.softmax(self.gating(x),dim=-2)
        encoded_space = torch.sum(weights*x,dim=1)
        result = self.classifier(encoded_space)
        if return_encoded_space:
            return result,encoded_space
        return result, None

if __name__ == "__main__":
    # Example usage
    embedding_size = 100
    layer_size = 42
    output_size = 1
    model = LayerFusion(layer_size,embedding_size)

    # Dummy input
    x = torch.randn(5, layer_size, embedding_size)
    output = model(x)
    print(output.shape)