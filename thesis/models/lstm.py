import torch.nn as nn

import wandb 
from matplotlib import pyplot as plt

# Define the LSTM Model with Initialization
class LSTMModel(nn.Module):
    def __init__(self, num_llm_layers,embedding_size, hidden_size,num_layers):
        super(LSTMModel, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_size, hidden_size,num_layers=num_layers, batch_first=True)
        
        # Fully connected layer (maps from hidden state output to the prediction)
        self.fc = nn.Linear(hidden_size, 1)
        
        self.aggregate = nn.Linear(num_llm_layers,1)
        
        self.activation = nn.ReLU()
        
        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize LSTM weights and biases
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:  # Input-hidden weights
                nn.init.xavier_uniform_(param.data)  # Xavier/Glorot initialization
            elif 'weight_hh' in name:  # Hidden-hidden weights
                nn.init.orthogonal_(param.data)  # Orthogonal initialization
        
        # Initialize fully connected layer
        nn.init.xavier_uniform_(self.fc.weight)  # Xavier initialization for the FC layer

    def forward(self, x, return_encoded_space=False):
        # Forward propagate LSTM
        out, _ = self.lstm(x) # h_t's, (h_n, c_n) 
        
            
        # Pass through classifier
        results = self.activation(self.fc(self.activation(out)))
        result = self.aggregate(results.squeeze(-1))
        if return_encoded_space:
            return result, out
        
        return result, None
    def plot_classifier_weights(self):
        aggregation_weights = self.aggregate.weight.data.squeeze(0)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(aggregation_weights)), aggregation_weights.cpu().numpy())
        ax.set_title('Aggregation Weights')
        ax.set_xlabel('Dimensions')
        ax.set_ylabel('Weight Value')
        wandb.log({"aggregation_weights":wandb.Image(fig)})