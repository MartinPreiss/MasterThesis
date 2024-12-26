
import torch.nn as nn
import torch
import torch.nn.functional as F
import wandb 
from matplotlib import pyplot as plt

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
        hidden_size = 100
        self.token_encoder = nn.Linear(embedding_size,hidden_size)
        self.layer_classifiers = nn.Conv1d(num_llm_layers,num_llm_layers,hidden_size)
        self.activation = nn.ReLU()
        self.aggregate = nn.Linear(num_llm_layers,1)

    def forward(self, x, return_encoded_space=False):
        encoded_space = self.activation(self.token_encoder(x))
        result =  self.aggregate(self.activation(self.layer_classifiers(encoded_space)).squeeze(dim=-1))
        if return_encoded_space:
            #contrastive loss might not work
            return result,encoded_space
        return result, None
    
    def plot_classifier_weights(self):
        weights = self.aggregate.weight.data.squeeze(0)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(weights)), weights.cpu().numpy())
        ax.set_title('Layer Weights')
        ax.set_xlabel('Dimensions')
        ax.set_ylabel('Weight Value')
        wandb.log({"last_layer_weights":wandb.Image(fig)})


class GatedLayerFusion(nn.Module):
    def __init__(self,num_llm_layers, embedding_size):
        super().__init__()
        self.token_encoder = nn.Linear(embedding_size,100)
        self.gating = nn.Conv1d(num_llm_layers,num_llm_layers,embedding_size)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(embedding_size,1)

    def forward(self, x,return_encoded_space=False):
        x = self.activation(self.token_encoder(x))
        weights = F.softmax(self.gating(x),dim=-2)
        encoded_space = torch.sum(weights*x,dim=1)
        result = self.classifier(encoded_space)
        if return_encoded_space:
            return result,encoded_space
        return result, None

    def plot_classifier_weights(self):
        print("weight print not possible (input dependent)")
        pass
    
class LayerFusionWithWeights(nn.Module):
    def __init__(self,num_llm_layers, embedding_size):
        super().__init__()
        #self.weights = nn.Parameter(torch.randn(num_llm_layers,embedding_size))
        self.weights = nn.Parameter(torch.randn(num_llm_layers,1))
        hidden_size = 100
        #linear layer_layer #embbeding_size -> embedding_size for every num_llm_layersnicolas alder
        self.token_encoder = nn.Linear(embedding_size,hidden_size)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(hidden_size,1)
        
    
    def forward(self, x,return_encoded_space=False):
        #x = [batch, num_layers, embedding_size]
        x = self.activation(self.token_encoder(x))
        
        encoded_space = torch.sum(self.weights*x,dim=1)
        result = self.classifier(self.activation(encoded_space))
        if return_encoded_space:
            return result,encoded_space
        return result, None
    
    
    def plot_classifier_weights(self):
        weights = self.weights.data.squeeze()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(weights)), weights.cpu().numpy()) 
        ax.set_title('Layer Weights')
        ax.set_xlabel('Dimensions')
        ax.set_ylabel('Weight Value')
        wandb.log({"layer_weights":wandb.Image(fig)})         
        
class EnsembleLayerFusionWithWeights(nn.Module):
    def __init__(self,num_llm_layers, embedding_size):
        super().__init__()
        
        self.num_llm_layers = num_llm_layers
        self.embedding_size = embedding_size
        
        self.weights = nn.Parameter(torch.randn(num_llm_layers,num_llm_layers,1))
        hidden_size = 100
        #linear layer_layer #embbeding_size -> embedding_size for every num_llm_layers
        self.token_encoder = nn.Linear(embedding_size,hidden_size)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(hidden_size,1)
        self.aggregate = nn.Linear(num_llm_layers,1)
        
    
    def forward(self, x,return_encoded_space=False):
        #x = [batch, num_layers, embedding_size
        
        #token encoder
        x = self.activation(self.token_encoder(x))
        
        #weighted sum to encapsulate layer relation 
        encoded_space = torch.sum(self.weights * x.unsqueeze(1), dim=2)
        
             
        
        #residual layer
        #encoded_space = encoded_space + x 
        
        #classify and aggregate ensemble 
        result = self.aggregate(self.activation(self.classifier(encoded_space)).squeeze(-1))
            
        if return_encoded_space:
            return result,encoded_space
        return result, None
    
    
    def plot_classifier_weights(self):
        weights = self.weights.data.view(self.num_llm_layers,self.num_llm_layers)
        
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
        sum_weights = weights.sum(dim=0).cpu().numpy()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(self.num_llm_layers), sum_weights)
        ax.set_title('Sum Fusion of All Layers Weights')
        ax.set_xlabel('Dimensions')
        ax.set_ylabel('Weight Value')
        wandb.log({"sum_fusion_weights":wandb.Image(fig)})
        
        aggregation_weights = self.aggregate.weight.data.squeeze(0)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(aggregation_weights)), aggregation_weights.cpu().numpy())
        ax.set_title('Aggregation Weights')
        ax.set_xlabel('Dimensions')
        ax.set_ylabel('Weight Value')
        wandb.log({"aggregation_weights":wandb.Image(fig)})
        
        
        
class LayerSimilarityClassifier(nn.Module):
    #idea get layer_realtion with cosine_similarity
    def __init__(self, num_llm_layers, embedding_size):
        super().__init__()
        
        self.num_llm_layers = num_llm_layers
        self.embedding_size = embedding_size
        
        print("TODO: Still no hidden size explaination")       
        hidden_size = 100
        self.token_encoder = nn.Linear(embedding_size,hidden_size)
        
        self.similarity = nn.CosineSimilarity(dim=-1)
        self.activation = nn.ReLU()
        self.final_classifier = nn.Linear(num_llm_layers*num_llm_layers,1)
        

    def forward(self, x, return_encoded_space=False):
        #encode_x 
        encoded_space = self.activation(self.token_encoder(x))
               
        #calculate layer_similarities
        x_normalized = nn.functional.normalize(encoded_space, dim=-1)
        similarities = torch.matmul(x_normalized, x_normalized.transpose(1, 2)).squeeze(-1) #-> [batch, num_layers, num_layers]
        
        #classify similarities 
        result = self.final_classifier(similarities.flatten(start_dim=1))
        
        """ 
        #--> for each layer independently 
        hidden_layer = self.activation(self.layer_classifier(similarities)).squeeze(-1)
        result = self.final_classifier(hidden_layer)
        """
        
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
        

class LayerAtentionClassifier(nn.Module):
    
    def __init__(self,num_llm_layers, embedding_size):
        super().__init__()
        
        self.num_llm_layers = num_llm_layers
        self.embedding_size = embedding_size
        hidden_size = 100
        
        #model architecture
        self.token_encoder = nn.Linear(embedding_size,hidden_size)
        self.attention_fusion = torch.nn.MultiheadAttention(hidden_size, 1,batch_first=True)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(num_llm_layers,1)
        self.aggregate = nn.Linear(num_llm_layers,1)
        self.final_classifier = nn.Linear(num_llm_layers*num_llm_layers,1)
        
    
    def forward(self, x,return_encoded_space=False):
        #x = [batch, num_layers, embedding_size]
        
        #token encoder
        x = self.activation(self.token_encoder(x))
        
        #attention fusion
        encoded_space, weights = self.attention_fusion(x,x,x)
        
        #calculate layer_similarities
        x_normalized = nn.functional.normalize(encoded_space, dim=-1)
        similarities = torch.matmul(x_normalized, x_normalized.transpose(1, 2)).squeeze(-1) #-> [batch, num_layers, num_layers]
        
        #classify similarities 
        result = self.final_classifier(similarities.flatten(start_dim=1))
        
        """ 
        #--> for each layer independently 
        hidden_layer = self.activation(self.layer_classifier(similarities)).squeeze(-1)
        result = self.final_classifier(hidden_layer)
        """
        
        if return_encoded_space:
            return result, encoded_space
        return result, None

        
        
        

if __name__ == "__main__":
    # Example usage
    embedding_size = 3584
    layer_size = 42
    batch_size = 100
    
    model = LayerAtentionClassifier(layer_size,embedding_size)

    # Dummy input
    x = torch.randn(batch_size, layer_size, embedding_size)
    output, _ = model(x)
    print(output.shape)
    #print(output)
    model.plot_classifier_weights()