# Hallucination Detection with the Internal Layers of LLMs

This repository contains the code and resources for the thesis:
**"Hallucination Detection with the Internal Layers of LLMs"**

## Abstract

Large Language Models (LLMs) have succeeded in a variety of natural language processing tasks [Zha+25]. However, they have notable limitations. LLMs tend to generate hallucinations, a seemingly plausible yet factually unsupported output [Hua+24], which have serious real-world consequences [Kay23; Rum+24]. Recent work has shown that probing-based classifiers that utilize LLMs’ internal representations can detect hallucinations [AM23; Bei+24; Bur+24; DYT24; Ji+24; SMZ24; Su+24]. This approach, since it does not involve model training, can enhance reliability without significantly increasing computational costs.

Building upon this approach, this thesis proposed novel methods for hallucination detection using LLM internal representations and evaluated them across three benchmarks: TruthfulQA, HaluEval, and ReFact. Specifically, a new architecture that dynamically weights and combines internal LLM layers was developed to improve hallucination detection performance. Throughout extensive experiments, two key findings were obtained: First, the proposed approach was shown to achieve superior performance compared to traditional probing methods, though generalization across benchmarks and LLMs remains challenging. Second, these generalization limitations were demonstrated to be mitigated through cross-benchmark training and parameter freezing. While not consistently improving, both techniques yielded better performance on individual benchmarks and reduced performance degradation when transferred to other benchmarks. These findings open new avenues for improving LLM reliability through internal representation analysis.

---
## Key Features

- Multiple model architectures for hallucination detection
- Configurable experiments via YAML files
- Tools for data processing, uncertainty estimation, and explainability
- Notebooks for analysis and visualization

---

## Project Structure

- `thesis/`: Main package containing code for data handling, model definitions, utilities, and configuration files.
	- `config/`: YAML configuration files for benchmarks, LLMs, models, and tasks.
	- `data_handling/`: Scripts for dataset creation, data processing, and prompt management.
	- `helpers/`: Utility scripts for metrics, model selection, and experiment management.
	- `models/`: Model architectures, including baselines, neural networks, and fusion models.
	- `ue/`: Uncertainty estimation and training scripts.
	- `xai/`: Explainability and analysis tools, including entropy analysis and logit lens.
	- `notebooks/`: Jupyter notebooks for data exploration and result visualization.
	- `shell_scripts/`: Shell scripts to automate experiments and evaluations.

---

## Getting Started

1. **Clone the repository**
	 ```sh
	 git clone <repo-url>
	 cd MasterThesis
	 ```

2. **Install dependencies**
	 - Using pipenv:
		 ```sh
		 pipenv install
		 pipenv shell
		 ```

3. **Configuration**
	 - Edit YAML files in `thesis/config/` to set up experiments, models, and datasets.

---

## Command Line Interface (CLI)

The main entry point is `thesis.__main__`, which provides a flexible CLI for running experiments and data processing tasks. You can specify the task and configuration via command line arguments or by editing the config files.

### Example Usage

```sh
python -m thesis --config-name=<your_config> task.name=<task_name>
```

#### Common Tasks

- **Create classification dataset:**
	```sh
	python -m thesis task.name=create_dataset
	```
- **Train layer fusion model:**
	```sh
	python -m thesis task.name=train_layer_fusion
	```
- **Test on benchmarks:**
	```sh
	python -m thesis task.name=test_on_benchmarks
	```

See the code and configs for more details.

---

## Citation

If you use this codebase, please cite the thesis:

```
Preiss, Martin. "Hallucination Detection with the Internal Layers of LLMs." Master’s Thesis, HPI, 2025.
```

---

## License

This project is for academic use. For other uses, please contact the author.