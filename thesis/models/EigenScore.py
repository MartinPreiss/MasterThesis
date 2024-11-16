import numpy as np
import torch 

# start copy and idea by https://github.com/alibaba/eigenscore/blob/main/func/metric.py#L174
def getEigenIndicator_v0(hidden_states): 
    alpha = 1e-3
    #Select all layers
    #selected_layer = int(len(hidden_states[0])/2)
    
    CovMatrix = torch.cov(hidden_states).cpu().numpy().astype(float)
    u, s, vT = np.linalg.svd(CovMatrix+alpha*np.eye(CovMatrix.shape[0]))
    eigenIndicator = np.mean(np.log10(s))
    return eigenIndicator, s


if __name__ == "__main__":  
    example_embeddings = torch.randn((42,4000))
    print(getEigenIndicator_v0(example_embeddings))