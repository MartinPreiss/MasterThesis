
import torch.nn as nn
import torch
import torch.nn.functional as F
import wandb 
from matplotlib import pyplot as plt



def no_comparison(x):
    # [batch, num_layers, input]
    # No comparison, just return the input
    return x

def dot_product(x):
    # [batch, num_layers, input]
    return torch.sum(x * x, dim=-1)

def euclidean_norm(x):
    # [batch, num_layers, input]
    return torch.linalg.norm(x,dim=-1)

def manhatten_norm(x):
    # [batch, num_layers, input]
    return torch.sum(x, dim=-1)

def pairwise_dot_product(x):
    # [batch, num_layers, input]
    # can be treated as a matrix multiplication 
    return torch.matmul(x, x.transpose(-1, -2))


def euclidean_distance(x):
    # [batch, num_layers, input]
    """naive but slow implementation
    pairwise_difference = x.unsqueeze(-2) - x.unsqueeze(-1)
    pairwise_distance = torch.linalg.norm(pairwise_difference, dim=-1)
   """
    return torch.cdist(x, x, p=2)

def manhatten_distance(x):
    # Compute pairwise Manhattan distance using broadcasting
    pairwise_distance = torch.cdist(x, x, p=1)
    return pairwise_distance
    

def cosine_similarity(x):
    # [batch, num_layers,embedding_size]
    x_normalized = F.normalize(x, p=2, dim=-1)
    similarities = torch.matmul(x, x_normalized.transpose(-1, -2))
    return similarities
    
    

class MLPEncoder(nn.Module):
    def __init__(self, input_size, activation_fun, num_layers=3):
        super(MLPEncoder, self).__init__()
        
        # List to hold the layers
        self.layers = nn.ModuleList()
        self.activation_func = activation_fun
        current_size = input_size
        
        # Dynamically create hidden layers
        for i in range(num_layers - 1):
            next_size = current_size // 2  # Halving the size each time
            self.layers.append(nn.Linear(in_features=current_size, out_features=next_size))
            
            current_size = next_size

        self.output_size = next_size
        
        print("no_init")
    
    def get_output_size(self):
        return self.output_size
    
    def forward(self, x):
        for layer in self.layers:
            x = self.activation_func(layer(x))
            
        return x

class SharedClassifierEnsemble(nn.Module):
    def __init__(self, num_llm_layers, input_size, output_size, activation_fun):
        super().__init__()
        
        self.num_llm_layers = num_llm_layers
        self.input_size = input_size
        self.output_size = output_size
        self.activation_fun = activation_fun
        
        self.classifier = nn.Linear(input_size, output_size)
        self.aggregate = nn.Linear(num_llm_layers, 1)

    def forward(self, x):
        #x = [batch, num_layers, num_layers]
        
        # Apply the classifier to each layer
        layer_outputs = self.activation_fun(self.classifier(x))
        #layer_outputs = [batch, num_layers, output_size]
        #--> view [batch, output_size, num_layers]
        layer_outputs = layer_outputs.transpose(-1, -2)
        result = self.aggregate(layer_outputs).squeeze(-1)
        
        return result
    
class DifferentClassifierEnsemble(nn.Module):
    def __init__(self, num_llm_layers, input_size, output_size, activation_fun):
        super().__init__()
        
        self.num_llm_layers = num_llm_layers
        self.input_size = input_size
        self.output_size = output_size
        self.activation_fun = activation_fun
        
        # Create a separate classifier for each layer, for parallel execution 
        self.layer_classifiers = nn.ModuleList([
            nn.Linear(input_size, output_size) for _ in range(num_llm_layers)
        ])
        # Each classifier will output a single value for each layer
        self.aggregate = nn.Linear(num_llm_layers, 1)

    def forward(self, x):
        #x = [batch, num_layers, input]
        
        # Apply the classifier to each layer
        layer_outputs = [self.activation_fun(classifier(x[:, i, :])) for i, classifier in enumerate(self.layer_classifiers)]
        
        layer_outputs  = torch.stack(layer_outputs,dim=1)  # [batch, num_layers, output_size]
        
        result = self.aggregate(layer_outputs.transpose(-1, -2)).squeeze(-1)  # [batch, output_size, num_layers]
        
        # Apply the classifier to each layer
        return result

class DirectClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.classifier = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        #x = [batch, input]
        result = self.classifier(x.flatten(-2,-1))
        return result

class LayerComparisionClassifier(nn.Module):
    
    def __init__(self, num_llm_layers, embedding_size, layer_depth,output_size, comparison_method, aggregation_method):
        super().__init__()
        
        self.num_llm_layers = num_llm_layers
        self.embedding_size = embedding_size
        self.layer_depth = layer_depth 
        self.comparison_method = comparison_method 
        
        self.name = f"LCC_{layer_depth}_{comparison_method}_{aggregation_method}"
    
        self.feature_extractor = MLPEncoder(embedding_size, nn.ReLU(), num_layers=layer_depth)
        
        encoded_size = self.feature_extractor.get_output_size()
        
        self.comparer = None
        aggregation_input_size = None
        nothing, single, pairwise = False, False, False
        if comparison_method == "no_comparison":
            self.comparer = no_comparison
            nothing = True
        elif comparison_method == "dot_product":
            self.comparer = dot_product
            single = True
        elif comparison_method == "manhatten":
            self.comparer = manhatten_norm
            single = True
        elif comparison_method == "euclidean_norm":
            self.comparer = euclidean_norm
            single = True
        elif comparison_method == "manhatten_distance":
            self.comparer = manhatten_distance
            pairwise = True
        elif comparison_method == "pairwise_dot_product":
            self.comparer = pairwise_dot_product
            pairwise = True
        elif comparison_method == "euclidean_distance":
            self.comparer = euclidean_distance
            pairwise = True
        elif comparison_method == "cosine":
            self.comparer = cosine_similarity
            pairwise = True
        
        
        self.aggregator = None
        if aggregation_method == "shared_classifier_ensemble":
            if nothing:
                aggregation_input_size = encoded_size
            elif single:
                raise Exception("Shared classifier ensemble cannot be used with single layer comparison methods")
            else: 
                aggregation_input_size = num_llm_layers
            self.aggregator = SharedClassifierEnsemble(num_llm_layers,aggregation_input_size, output_size, nn.ReLU())
            
        elif aggregation_method == "different_classifiers_ensemble":
            if comparison_method == "no_comparison":
                aggregation_input_size = encoded_size
            elif single:
                raise Exception("Different classifier ensemble cannot be used with single layer comparison methods")
            else: 
                aggregation_input_size = num_llm_layers
            self.aggregator = DifferentClassifierEnsemble(num_llm_layers,aggregation_input_size , output_size, nn.ReLU())
        elif aggregation_method == "flattend_aggregation":
            
            if comparison_method == "no_comparison":
                aggregation_input_size = encoded_size * num_llm_layers
            elif single:
                aggregation_input_size = num_llm_layers
            else: 
                aggregation_input_size = num_llm_layers * num_llm_layers
            self.aggregator = DirectClassifier(aggregation_input_size, output_size)
        
        
        if self.comparer is None:
            raise ValueError(f"Invalid comparison method: {comparison_method}")
        if self.aggregator is None:
            raise ValueError(f"Invalid aggregation method: {aggregation_method}")
        

    def forward(self, x, return_encoded_space=False):
        #encode_x 
        encoded_space = self.feature_extractor(x)
        
        comparison = self.comparer(encoded_space)
        
        result = self.aggregator(comparison)
        
        if return_encoded_space:
            #contrastive loss might not work
            return result, encoded_space
        return result, None

    def plot_classifier_weights(self):
        
        weights = self.final_classifier.weight.data.squeeze(0).view(self.num_llm_layers,self.num_llm_layers)
        
        # Create a figure for all layers with each layer having a subplot
        fig, axes = plt.subplots(self.num_llm_layers, 1, figsize=(10, 2 * self.num_llm_layers))
        for i in range(self.num_llm_layers):
            axes[i].bar(range(self.num_llm_layers), weights[i].cpu().numpy())
            axes[i].set_title(f'Layer {i} Weights')
            axes[i].set_xlabel('Dimensions')
            axes[i].set_ylabel('Weight Value')
        plt.tight_layout()
        wandb.log({"all_layers_weights":wandb.Image(fig)})

        # Create a figure for the sum fusion of all layers
        sum_weights = torch.abs(weights).sum(dim=0).cpu().numpy()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(self.num_llm_layers), sum_weights)
        ax.set_title('Sum Fusion of All Layers Weights')
        ax.set_xlabel('Dimensions')
        ax.set_ylabel('Weight Value')
        wandb.log({"sum_fusion_weights":wandb.Image(fig)})
    
    def freeze_last_layers(self):
        self.aggregator.requires_grad = False


if __name__ == "__main__":
    # Example usage
    
    # Example usage
    depth = 3
    embedding_size = 3584
    layer_size = 42
    batch_size = 100
    output_size = 5
    
    comparison_methods = ["no_comparison", "dot_product", "euclidean_norm", "manhatten", "pairwise_dot_product", "euclidean_distance", "manhatten_distance", "cosine"]
    aggregation_methods = ["shared_classifier_ensemble", "different_classifiers_ensemble", "flattend_aggregation"]
    
    for comparison_method in comparison_methods:
        x = torch.randn(batch_size, layer_size, embedding_size)
        if comparison_method == "cosine":
            comparer = cosine_similarity
        elif comparison_method == "dot_product":
            comparer = dot_product
        elif comparison_method == "manhatten":
            comparer = manhatten_norm
        elif comparison_method == "euclidean_norm":
            comparer = euclidean_norm
        elif comparison_method == "manhatten_distance":
            comparer = manhatten_distance
        elif comparison_method == "pairwise_dot_product":
            comparer = pairwise_dot_product
        elif comparison_method == "euclidean_distance":
            comparer = euclidean_distance
        elif comparison_method == "no_comparison":
            comparer = no_comparison
        comparison = comparer(x)
        print(f"Comparison method: {comparison_method}, Output shape: {comparison.shape}")
    """
    for comparison_method in comparison_methods:
        for aggregation_method in aggregation_methods:
            model = LayerComparisionClassifier(layer_depth=depth, 
                                               embedding_size=embedding_size, 
                                               num_llm_layers=layer_size, 
                                               output_size=output_size, 
                                               comparison_method=comparison_method, 
                                               aggregation_method=aggregation_method)
            # Dummy input
            x = torch.randn(batch_size, layer_size, embedding_size)
            output, _ = model(x)
            print(output.shape)
         
    model = LayerComparisionClassifier(layer_depth=depth, 
                                               embedding_size=embedding_size, 
                                               num_llm_layers=layer_size, 
                                               output_size=output_size, 
                                               comparison_method="euclidean", 
                                               aggregation_method="shared_classifier_ensemble")
    
    # Dummy input
    x = torch.randn(batch_size, layer_size, embedding_size)
    output, _ = model(x)
    print(output.shape)
    """
    