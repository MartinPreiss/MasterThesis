import torch
import torch.nn as nn
import torch.optim as optim

# Define the LSTM Model with Initialization
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers):
        super(LSTMModel, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size,num_layers=num_layers, batch_first=True)
        
        # Fully connected layer (maps from hidden state output to the prediction)
        self.fc = nn.Linear(hidden_size, 1)
        
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

    def forward(self, x):
        # Forward propagate LSTM
        out, _ = self.lstm(x) # h_t's, (h_n, c_n) 
        #out has shape of [batch,sequence_length,hidden_size]
            
        # Pass through fully connected layer
        out = self.fc(out[:,-1,:])
        
        return out
