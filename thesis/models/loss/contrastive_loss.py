import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        """
        Initializes the ContrastiveLoss module.
        
        Args:
            margin (float): The margin for the contrastive loss.
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, binary_labels):
        #embedings shape [batch_size, encoded_space] 
        #binary labels shape [batch_size,classes]
        
        #get_positive / negative indices 
        
        labels_flat = binary_labels.squeeze(1)  # Convert (batch, 1) to (batch,)

        # Step 2: Create a mask for labels == 1
        positive_mask = labels_flat == 1
        negative_mask = labels_flat == 0
        
        negative_embeddings = embeddings[negative_mask]
        
                
        #sample 1 positive anchor of batch
        #sample another positve  of batch 
        positive_indices = torch.nonzero(positive_mask)
        if len(positive_indices) > 1:  # Ensure there are positive samples
            positive_indices = positive_indices[torch.randperm(len(positive_indices))[:2]]
            anchor = embeddings[positive_indices[0]]
            positive_sample = embeddings[positive_indices[1]]
        else:
            print("Not 2 positive labels found.")
            return 0

        
        #similiarity of 2 positives 
        positive_similiarity = torch.nn.CosineSimilarity()(anchor,positive_sample)
        
        negative_similarities = torch.nn.functional.cosine_similarity(
            anchor, negative_embeddings
        )  # Shape: [num_negatives]
                
        loss = -1*torch.log(torch.exp(positive_similiarity/0.1) / torch.exp(negative_similarities/0.1).sum())
        #exponent on  dot product results of j
        
        #normalize division
        
        #logarithm
        
        return loss
        
        
        
        
        
        
        #chat gpt implementation:
        """
        Forward pass for the contrastive loss.
        
        Args:
            embeddings (torch.Tensor): Tensor of shape (batch_size, embedding_dim).
            binary_labels (torch.Tensor): Tensor of shape (batch_size,) with 0 or 1.
        
        Returns:
            torch.Tensor: The computed contrastive loss.
        
        # Compute pairwise distances between embeddings
        pairwise_distances = torch.cdist(embeddings, embeddings, p=2)  # shape: (batch_size, batch_size)

        # Expand labels for pairwise computation
        labels = binary_labels.unsqueeze(1)  # shape: (batch_size, 1)
        label_matrix = (labels == labels.T).float()  # shape: (batch_size, batch_size)

        # Contrastive loss components
        positive_loss = label_matrix * pairwise_distances.pow(2)
        negative_loss = (1 - label_matrix) * torch.clamp(self.margin - pairwise_distances, min=0).pow(2)

        # Combine the losses
        loss = positive_loss + negative_loss

        # Compute the mean loss
        return loss.mean()
        """

if __name__ == "__main__":
    # Example usage
    embeddings = torch.randn(20, 128)  # Example embeddings (batch_size=8, embedding_dim=128)
    binary_labels = torch.randint(0, 2, (20,))  # Example binary labels (0 or 1)
    binary_labels = binary_labels.unsqueeze(1)

    criterion = ContrastiveLoss(margin=1.0)
    loss = criterion(embeddings, binary_labels)

    print(f"Contrastive Loss: {loss.item()}")
