from thesis.models.neural_net import AllLayerClassifier, GatedLayerFusion, MLP, LayerFusionWithWeights, LayerSimilarityClassifier,EnsembleLayerFusionWithWeights, LayerAtentionClassifier, Baseline1, Baseline2, EuclideanDistanceClassifier
from thesis.models.lstm import LSTMModel
from thesis.models.mamba import MambaClassifier
from thesis.models.layer_comparison_classifier import LayerComparisonClassifier, LCC_with_CRF

def get_model(cfg,embedding_size, num_layers):
    if cfg.model.name == "all_layer_classifier":
        return AllLayerClassifier(embedding_size=embedding_size, num_llm_layers=num_layers)
    elif cfg.model.name == "gated_layer_fusion":
        return GatedLayerFusion(embedding_size=embedding_size, num_llm_layers=num_layers)
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
    elif cfg.model.name == "ensemble_weight_fusion":
        return EnsembleLayerFusionWithWeights(embedding_size=embedding_size, num_llm_layers=num_layers)
    elif cfg.model.name == "layer_attention":  
        return LayerAtentionClassifier(embedding_size=embedding_size, num_llm_layers=num_layers)
    elif cfg.model.name == "baseline1":
        return Baseline1(embedding_size=embedding_size, num_llm_layers=num_layers)
    elif cfg.model.name == "baseline2":
        return Baseline2(embedding_size=embedding_size, num_llm_layers=num_layers)
    elif cfg.model.name == "euclidean_distance":
        return EuclideanDistanceClassifier(embedding_size=embedding_size, num_llm_layers=num_layers)
    elif cfg.model.name == "layer_comparison_classifier":
        return  LayerComparisonClassifier(embedding_size=embedding_size, 
                                          num_llm_layers=num_layers, 
                                          output_size=cfg.model.num_classes, 
                                          layer_depth=cfg.model.layer_depth,
                                          comparison_method=cfg.model.comparison_method, 
                                          aggregation_method=cfg.model.aggregation_method,
                                          final_classifier_non_linear=cfg.model.final_classifier_non_linear)
    
    elif cfg.model.name == "lcc_with_crf":
        return  LCC_with_CRF(embedding_size=embedding_size, 
                                          num_llm_layers=num_layers, 
                                          output_size=cfg.model.num_classes, 
                                          layer_depth=cfg.model.layer_depth,
                                          comparison_method=cfg.model.comparison_method, 
                                          aggregation_method=cfg.model.aggregation_method,
                                          final_classifier_non_linear=cfg.model.final_classifier_non_linear,
                                          num_classes=cfg.model.num_classes,
                                        )