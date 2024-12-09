
import hydra
from omegaconf import DictConfig, OmegaConf

from thesis.ue.train_per_layer import train_per_layer
from thesis.ue.continue_learning import continue_learning
from thesis.ue.train_log_reg import train_log_reg
from thesis.ue.train_layer_fusion import train_layer_fusion
from thesis.data_handling.create_dataset import create_dataset

from thesis.utils import print_cuda_info


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    options = {
        "train_per_layer": train_per_layer,
        "continue_learning":continue_learning,
        "train_log_reg": train_log_reg,
        "train_layer_fusion":train_layer_fusion,
        "create_dataset": create_dataset,
        "playground": playground
    }

    print_cuda_info()
    options[cfg.task.name](cfg)

def playground(cfg):
    pass

if __name__ == "__main__":
    main()