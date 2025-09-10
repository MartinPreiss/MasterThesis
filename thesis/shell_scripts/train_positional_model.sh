#!/bin/bash

#nohup /train_all_fusion_models.sh &> train_all_fusion_models.log & disown

# Display current directory
pwd

# Activate pipenv shell
pipenv shell

echo "PID: $!" 


#potential options use_pretrained, use_downsampling  use_pipeline: False freeze_lcc: False weighting off 
# model no_comparison 
# but with crf and not

# i need to see runs without any saving methods 
# need to see wether weighting does something 
# need to see wether crf can learn anyhting freezed or not  
# is pretrained important ? should i discard ? 


# lcc runs only  
# lcc with pretrained 
# lcc with weighitng 
# lcc with downsampling 
# lcc with all together 
# lcc with pretrained and weighting 

# python -m thesis task=train_positional_layer_fusion wandb.name=llc_only wandb.use_wandb=True model=layer_comparison_classifier model.comparison_method=no_comparison  task.training_params.use_cross_entropy_weighting=False task.use_pretrained=False  task.use_downsampling=False
# 
# python -m thesis task=train_positional_layer_fusion wandb.name=llc_only_use_pretrained wandb.use_wandb=True model=layer_comparison_classifier model.comparison_method=no_comparison  task.training_params.use_cross_entropy_weighting=False task.use_pretrained=True  task.use_downsampling=False
# 
# python -m thesis task=train_positional_layer_fusion wandb.name=llc_only_use_weight wandb.use_wandb=True model=layer_comparison_classifier model.comparison_method=no_comparison  task.training_params.use_cross_entropy_weighting=True task.use_pretrained=False  task.use_downsampling=False
# 
# 
# python -m thesis task=train_positional_layer_fusion wandb.name=llc_only_use_downsampling wandb.use_wandb=True model=layer_comparison_classifier model.comparison_method=no_comparison  task.training_params.use_cross_entropy_weighting=False task.use_pretrained=False  task.use_downsampling=True
# 
# python -m thesis task=train_positional_layer_fusion wandb.name=llc_only_all wandb.use_wandb=True model=layer_comparison_classifier model.comparison_method=no_comparison  task.training_params.use_cross_entropy_weighting=True task.use_pretrained=True  task.use_downsampling=True
# 
# python -m thesis task=train_positional_layer_fusion wandb.name=llc_only_all_no_downsampling wandb.use_wandb=True model=layer_comparison_classifier model.comparison_method=no_comparison  task.training_params.use_cross_entropy_weighting=True task.use_pretrained=True  task.use_downsampling=False


# crf runs 
# crf only 
# crf only with use_cross_entropy_loss + weighting ? 
# crf with pretrained 

# pipeline runs 
python -m thesis task=train_positional_layer_fusion wandb.name=pipeline_no_comparison_freezed wandb.use_wandb=True model=lcc_with_crf model.comparison_method=no_comparison task.use_pipeline=True task.freeze_lcc=True

python -m thesis task=train_positional_layer_fusion wandb.name=pipeline_no_comparison_not_freezed wandb.use_wandb=True model=lcc_with_crf model.comparison_method=no_comparison task.use_pipeline=True task.freeze_lcc=False

python -m thesis task=train_positional_layer_fusion wandb.name=crf_only wandb.use_wandb=True model=lcc_with_crf model.comparison_method=no_comparison  task.training_params.use_cross_entropy_weighting=False task.use_pretrained=False  task.use_downsampling=False

# python -m thesis task=train_positional_layer_fusion wandb.name=no_comparison_runs wandb.use_wandb=True model=layer_comparison_classifier model.comparison_method=no_comparison  task.training_params.use_cross_entropy_weighting=True
# 
# 
# python -m thesis task=train_positional_layer_fusion wandb.name=no_comparison_runs wandb.use_wandb=True model=lcc_with_crf model.comparison_method=no_comparison
# 
# python -m thesis task=train_positional_layer_fusion task.use_pipeline=True  wandb.use_wandb=True  wandb.name=pipeline_no_comparison model=lcc_with_crf model.comparison_method=no_comparison task.freeze_lcc=False
# 
# python -m thesis task=train_positional_layer_fusion task.use_pipeline=True wandb.name=pipeline_no_comparison_freezed wandb.use_wandb=True model=lcc_with_crf model.comparison_method=no_comparison task.freeze_lcc=True

