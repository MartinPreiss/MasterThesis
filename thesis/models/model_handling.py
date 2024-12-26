from thesis.models.neural_net import AllLayerClassifier, LayerFusion, MLP, LayerFusionWithWeights, LayerSimilarityClassifier
from thesis.models.lstm import LSTMModel
from thesis.models.mamba import MambaClassifier

def get_model(cfg,embedding_size, num_layers):
    if cfg.model.name == "all_layer_classifier":
        return AllLayerClassifier(embedding_size=embedding_size, num_llm_layers=num_layers)
    elif cfg.model.name == "layer_fusion":
        return LayerFusion(embedding_size=embedding_size, num_llm_layers=num_layers)
    elif cfg.model.name == "lstm":
        return LSTMModel(embedding_size=embedding_size, num_llm_layers=num_layers,hidden_size=cfg.model.hidden_size,num_layers=cfg.model.num_layers)
    elif cfg.model.name == "mamba":
        return MambaClassifier(embedding_size,cfg.model.num_hidden_layers)
    elif cfg.model.name == "mlp":
        return MLP(embedding_size,num_layers=cfg.model.num_layers)
    elif cfg.model.name == "layer_fusion_weights":
        return LayerFusionWithWeights(embedding_size=embedding_size, num_llm_layers=num_layers)
    elif cfg.model.name == "layer_similarity_classifier":
        return LayerSimilarityClassifier(embedding_size=embedding_size, num_llm_layers=num_layers)