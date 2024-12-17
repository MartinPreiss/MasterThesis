import torch
import torch.nn as nn



# old class which maybe can be removed 
# was just the first try of implementing the contrastive loss
class ContrastiveLossInternalInspectorPaper(nn.Module):
    def __init__(self, margin=1.0):
        """
        Initializes the ContrastiveLoss module.
        
        Args:
            margin (float): The margin for the contrastive loss.
        """
        super(ContrastiveLossInternalInspectorPaper, self).__init__()
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


class ContrastiveLoss(nn.Module):
    ## https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/ContrastiveLoss.py 
    def __init__(
        self,
        distance_metric= nn.CosineSimilarity(),
        margin: float = 0.5,
        size_average: bool = True,
    ) -> None:
        
        super().__init__()
        self.distance_metric = distance_metric
        self.margin = margin
        self.size_average = size_average

    def forward(self,  embeddings, binary_labels):
        # Contrastive loss. Expects as input two texts and a label of either 0 or 1. If the label == 1, then the distance between the
        # two embeddings is reduced. If the label == 0, then the distance between the embeddings is increased.
        
        
        # ---> have to build pairs of anchor and other 
        # ---> get relation of anchor and other (either two positive pair/ two netative pair reduece distance, if two differents increase distance)

        # Generate a random permutation of indices
        permuted_indices = torch.randperm(embeddings.size(0))

        # Split the permuted indices into pairs
        anchor_indices = permuted_indices[:embeddings.size(0) // 2]
        other_indices = permuted_indices[embeddings.size(0) // 2:]
        # Get the embeddings for anchors and others
        anchor_embeddings = embeddings[anchor_indices]
        other_embeddings = embeddings[other_indices]

        # Get the corresponding binary labels for the pairs
        anchor_labels = binary_labels[anchor_indices]
        other_labels = binary_labels[other_indices]
        
        labels = ((anchor_labels + other_labels) + 1) % 2 # 0 if both labels are the different, 1 if they are same
        
        #get representations of anchors and others with binary_labels class 
        
        #get_anchors and counterparts
        
        distances = self.distance_metric(anchor_embeddings, other_embeddings)
        losses = 0.5 * (
            labels.float() * distances.pow(2) + (1 - labels).float() * nn.functional.relu(self.margin - distances).pow(2)
        )
        return losses.mean() if self.size_average else losses.sum()

if __name__ == "__main__":
    # Example usage
    embeddings = torch.randn(20, 128)  # Example embeddings (batch_size=8, embedding_dim=128)
    binary_labels = torch.randint(0, 2, (20,))  # Example binary labels (0 or 1)
    binary_labels = binary_labels.unsqueeze(1)

    criterion = ContrastiveLoss()
    loss = criterion(embeddings, binary_labels)

    print(f"Contrastive Loss: {loss.item()}")
