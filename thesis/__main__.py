
import hydra
from omegaconf import DictConfig, OmegaConf

from thesis.ue.train_classifier import train_classifier
from thesis.data_handling.create_dataset import create_dataset


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    options = {
        "train_classifier": train_classifier,
        "create_dataset": create_dataset
    }

    options[cfg.task.name](cfg)

if __name__ == "__main__":
    main()