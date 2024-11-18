
import hydra
from omegaconf import DictConfig, OmegaConf

from thesis.ue.train_classifier import train_classifier
from thesis.ue.train_lstm import train_lstm
from thesis.ue.train_log_reg import train_log_reg
from thesis.ue.train_all_layer_classifier import train_all_layer_classifier
from thesis.data_handling.create_dataset import create_dataset

from thesis.utils import print_cuda_info


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    options = {
        "train_classifier": train_classifier,
        "train_lstm": train_lstm,
        "train_log_reg": train_log_reg,
        "train_all_layer_classifier":train_all_layer_classifier,
        "create_dataset": create_dataset
    }

    print_cuda_info()
    options[cfg.task.name](cfg)

if __name__ == "__main__":
    main()