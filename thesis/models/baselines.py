import torch.nn as nn
import torch
import torch.nn.functional as F
from thesis.models.layer_comparison_classifier import CRFAggregator


class SingleLayer(nn.Module):
    def __init__(self, num_llm_layers, embedding_size, layer_depth,output_size,target_layer):
        super().__init__()
        
        self.num_llm_layers = num_llm_layers
        self.embedding_size = embedding_size
        
        current_size = embedding_size
        self.layers = nn.ModuleList()
        self.target_layer = target_layer
        
        # Dynamically create hidden layers
        for i in range(layer_depth-1):
            next_size = current_size // 2  # Halving the size each time
            self.layers.append(nn.Linear(in_features=current_size, out_features=next_size))
            current_size = next_size

        # Final layer
        self.layers.append(nn.Linear(in_features=current_size, out_features=output_size))
        self.activation_func = nn.ReLU() 

        
    def forward(self,x,user_contrastive=False): 
        # x shape = [batch, num_layers, embedding_size]
        x = x[:, self.target_layer, :].squeeze(1)  # Select the middle layer

        for layer in self.layers[:-1]:
            x = self.activation_func(layer(x))
        x = self.layers[-1](x)
        return x, None

    
class MiddleLayer(SingleLayer):
    
    def __init__(self, num_llm_layers, embedding_size, layer_depth,output_size):
        print("hardcoded layer depth")
        layer_depth = 2 
        super().__init__(num_llm_layers, embedding_size, layer_depth,output_size,num_llm_layers // 2)
        
        self.target_layer = self.num_llm_layers // 2
            
class LastLayer(SingleLayer):
    def __init__(self, num_llm_layers, embedding_size, layer_depth,output_size):
        print("hardcoded layer depth")
        layer_depth = 3 
        super().__init__(num_llm_layers, embedding_size, layer_depth,output_size,-1)
        
        self.target_layer = -1  # Last layer

class StackedLayers(nn.Module):
    
    def __init__(self, num_llm_layers, embedding_size, layer_depth,output_size):
        print("hardcoded layer depth")
        layer_depth = 3
        super().__init__()
        
        self.num_llm_layers = num_llm_layers
        self.embedding_size = embedding_size
        
        current_size = embedding_size * num_llm_layers
        self.layers = nn.ModuleList()
        
        # Dynamically create hidden layers
        for i in range(layer_depth-1):
            if i == 0:
                next_size = embedding_size
            else:
                next_size = current_size // 2  # Halving the size each time
            self.layers.append(nn.Linear(in_features=current_size, out_features=next_size))
            current_size = next_size

        # Final layer
        self.layers.append(nn.Linear(in_features=current_size, out_features=output_size))
        self.activation_func = nn.ReLU() 

        
        
    def forward(self,x,user_contrastive=False): 
        # x shape = [batch, num_layers, embedding_size]
        x = x.view(x.size(0), -1)  # Flatten the input to [batch, num_layers * embedding_size]
        for layer in self.layers[:-1]:
            x = self.activation_func(layer(x))
        x = self.layers[-1](x)
        return x, None

class AllLayersEnsemble(nn.Module):
    def __init__(self, num_llm_layers, embedding_size, layer_depth,output_size):
        print("hardcoded layer depth")
        layer_depth = 1 
        super().__init__()
        
        self.num_llm_layers = num_llm_layers
        self.embedding_size = embedding_size
    
        self.layer_classifier = nn.ModuleList()
        
        # Dynamically create hidden layers
        for i in range(num_llm_layers): 
            self.layer_classifier.append(SingleLayer(num_llm_layers, embedding_size, layer_depth,output_size,i))
        # Final layer
        self.aggregation_layer= nn.Linear(in_features=num_llm_layers, out_features=1)
        self.activation_func = nn.ReLU()     
    def forward(self,x,user_contrastive=False): 
        outs = torch.stack([layer(x)[0] for layer in self.layer_classifier])  # Shape: [batch, num_llm_layers, output_size]
        outs = outs.permute(1, 2, 0)  # Change to [batch, output_size, num_llm_layers]
        result = self.aggregation_layer(self.activation_func(outs)).squeeze(-1)
        return result, None

class BaselineWithCrf(nn.Module):
        
    def __init__(self, cfg,num_llm_layers, embedding_size):
        super().__init__()
        self.num_llm_layers = num_llm_layers
        self.embedding_size = embedding_size
        old_name = cfg.model.name
        cfg.model.name = cfg.model.classifier_name
        from thesis.models.model_handling import get_model
        self.classifier  = get_model(cfg,embedding_size,num_llm_layers)
        self.crf = CRFAggregator(cfg.model.num_classes)
        cfg.model.name = old_name

    def forward(self, x,contrastive_loss=False, tags=None, mask=None):
        #x should be [batch, seq_length, num_layers, embedding_size]
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        x = x.view(batch_size * seq_length, self.num_llm_layers, self.embedding_size)
        result, _ = self.classifier(x,contrastive_loss)
        # Reshape result to [batch_size, seq_length, classes]
        result = result.view(batch_size, seq_length, -1)

        return self.crf(result, tags, mask) 

if __name__ == "__main__":
    # Example usage
    num_llm_layers = 12
    embedding_size = 768
    layer_depth = 4
    batch_size = 32
    output_size = 5

    x = torch.randn(batch_size, num_llm_layers, embedding_size)
    model = StackedLayers(num_llm_layers, embedding_size, layer_depth, output_size)
    output, _ = model(x)
    print(output.shape)  # Should be [batch_size, output_size]

    model = LastLayer(num_llm_layers, embedding_size, layer_depth, output_size)
    output, _ = model(x)
    print(output.shape)  # Should be [batch_size, output_size] 

    model = MiddleLayer(num_llm_layers, embedding_size, layer_depth, output_size)
    output, _ = model(x)
    print(output.shape)

    model = AllLayersEnsemble(num_llm_layers, embedding_size, layer_depth, output_size)
    output, _ = model(x)
    print(output.shape)