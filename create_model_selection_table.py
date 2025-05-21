import torch
import os
import pandas as pd
import numpy as np

comparison_mapping = {
    "no_comparison": "No Comparison",
    "dot_product": "Dot Product",
    "euclidean_norm": "Euclidean Norm",
    "manhatten": "Manhatten Norm",
    "pairwise_dot_product": "Pairwise Dot Product",
    "euclidean_distance": "Euclidean Distance",
    "manhatten_distance": "Manhatten Distance",
    "cosine": "Cosine Similarity"
}

aggregation_mapping = {
    "shared_classifier_ensemble_True": "Shared Classifier Ensemble (Non-Linear)",
    "shared_classifier_ensemble_False": "Shared Classifier Ensemble (Linear)",
    "flattend_aggregation_True": "Flattened Aggregation (Non-Linear)",
    "flattend_aggregation_False": "Flattened Aggregation (Linear)"
}

def create_baseline_table(path, layer_depths):
    baseline_names = ["last_layer", "middle_layer", "stacked_layers", "all_layers_ensemble"]
    baseline_mapping = {
        "middle_layer": "MiddleLayer",
        "last_layer": "LastLayer",
        "stacked_layers": "StackedLayers",
        "all_layers_ensemble": "AllLayersEnsemble"
    }
    baseline_f1_scores = {name: {depth: None for depth in layer_depths} for name in baseline_names}

    for layer_depth in layer_depths:
        for baseline_name in baseline_names:

            file_name = f"{baseline_name}_haluleval_{layer_depth}_5_f1s.pth"
            file_path = os.path.join(path, file_name)
            
            if os.path.exists(file_path):
                baseline_f1_scores[baseline_name][layer_depth] = torch.load(file_path).mean().item()
            else:
                print(f"File not found: {file_path}")
                continue
    # Generate LaTeX table
    latex_table = r"""
\begin{table}[h]
\begin{tabular}{|l|lllll|}
\hline
\multicolumn{1}{|c|}{\textbf{Baseline}} & \multicolumn{5}{c|}{\textbf{Layer Depth}} \ \cline{2-6}
\multicolumn{1}{|c|}{} & \multicolumn{1}{l|}{1} & \multicolumn{1}{l|}{2} & \multicolumn{1}{l|}{3} & \multicolumn{1}{l|}{4} & \multicolumn{1}{l|}{5} \ \hline
"""
    for name, depths in baseline_f1_scores.items():
        latex_table += f"{baseline_mapping[name]} & " + " & ".join(
            [f"{depths[depth]:.2f}" if depths[depth] is not None else "" for depth in layer_depths]
        ) + r" \\ \hline" + "\n"
    latex_table += r"""
\end{tabular}
\caption{F1 Scores of different baseline approaches with multiple layer depths for the text classification task}
\label{tab:baseline_f1_scores}
\end{table}
"""
    return latex_table      

def create_llm_table(path, comparison_method,benchmark_name="truthfulqa"):
    llms = ["meta-llama/Llama-3.1-8B-Instruct" ,"google/gemma-3-1b-it" ,"google/gemma-3-4b-it" ,"google/gemma-3-12b-it" ,"google/gemma-3-27b-it" ,"meta-llama/Llama-3.2-1B-Instruct" ,"meta-llama/Llama-3.2-3B-Instruct" ,"meta-llama/Llama-3.3-70B-Instruct"]
    llms = [llm[llm.rfind("/")+1:] for llm in llms]
    latex_rows = []
    for llm in llms:
        # example file layer_comparison_classifier_refact_gemma-3-12b-it_1_no_comparison_flattend_aggregation_False_2_False_5_f1s.pth 
        file_name = f"layer_comparison_classifier_{benchmark_name}_{llm}_1_{comparison_method}_flattend_aggregation_False_2_False_5"
        test_file_path = os.path.join(path, file_name+"_f1s.pth")
        val_file_path = os.path.join(path, file_name+"_val_f1s.pth")
        if os.path.exists(test_file_path):
            test_f1_scores = torch.load(test_file_path)
            val_f1_scores = torch.load(val_file_path)
            
            #filter out f1 scores that are zero 
            original_mean = test_f1_scores.mean().item()
            filtered_count = val_f1_scores[val_f1_scores == 0].shape[0]
            test_f1_scores = test_f1_scores[val_f1_scores != 0]
            filtered_mean = test_f1_scores.mean().item()

            latex_rows.append(f"{llm} & {original_mean:.2f} & {filtered_mean:.2f} & {filtered_count} \\\\ \\hline") 
        else:
            #print(f"File not found: {file_path}")
            latex_rows.append(f"{llm} & N/A & N/A & N/A \\\\ \\hline")

            continue
    # Generate LaTeX table
    latex_table = r"""
\begin{table}[]
\begin{tabular}{|l|l|l|l|}
\hline
\multicolumn{1}{|c|}{\textbf{LLM}} & \textbf{F1 Score} & \textbf{Filtered F1 Score} & \textbf{Filtered Count} \\ \hline
"""
    for latex_row in latex_rows:
        latex_table += latex_row + "\n"
    latex_table += r"""
\end{tabular}
\caption{F1 Scores for Different LLMs using Comparison Method: """ + comparison_mapping[comparison_method] + r"""}
\label{tab:llm_f1_scores}
\end{table}
"""
    return latex_table

def create_in_domain_shift_table(comparison_method, pretrained=True, freeze=True):
    # thesis/data/avg_in_domain_shift/haluleval_refact_results.pth
    if pretrained:
        if freeze:
            path = "./thesis/data/avg_in_domain_shift_pretrained_original_hyp"
        else:
            path = "./thesis/data/avg_in_domain_shift_pretrained_no_freeze"
    else:
        path =  "./thesis/data/avg_in_domain_shift_original_hyp"
        #path = "./thesis/data/avg_in_domain_shift"
    files = os.listdir(path) 
    latex_rows = []
    all_test_scores = []
    benchmark_names  = ["refact", "haluleval", "truthfulqa"]
    for file in files: 
        if not comparison_method in file: 
            continue 
        result_dict = torch.load(os.path.join(path, file))
        test_results = result_dict["test_results"]
        val_results = result_dict["val_results"]
        #postprocess the f1 score dict 
        val_f1_scores = torch.tensor([val_score_dict["f1"] for val_score_dict in val_results])
        
        benchmark_test_f1_scores = []
        filtered_count = val_f1_scores[val_f1_scores == 0].shape[0]

        benchmark_results = []
        for benchmark_name in benchmark_names:
            test_f1_scores = torch.tensor([test_score_dict[benchmark_name]["f1"] for test_score_dict in test_results])
            all_test_scores.append(test_f1_scores)
            #filter out f1 scores that are zero
            original_mean = test_f1_scores.mean().item()
            test_f1_scores = test_f1_scores[val_f1_scores != 0]
            filtered_mean = test_f1_scores.mean().item()
            benchmark_results.append((original_mean, filtered_mean))
            
        if pretrained: 
            first_benchmark_name = file.split("_")[0]
            second_benchmark_name = file.split("_")[1]

            results_concatenated = " & ".join([f"{test_result[1]:.2f}" for test_result in benchmark_results])
            latex_row = f"{first_benchmark_name} & {second_benchmark_name} & {results_concatenated} & {filtered_count} \\\\ \\hline"

        else:
            second_benchmark_name = file.split("_")[1]
            benchmark_results = [f"{test_result[1]:.2f}" for test_result in benchmark_results]
            latex_row = f"{second_benchmark_name} & " + " & ".join(benchmark_results) + f" & {filtered_count} \\\\ \\hline"
        latex_rows.append(latex_row)

    print("total avg f1 score", torch.cat(all_test_scores).mean().item())
    #-----> all cosine method
    # total avg of 0.2793 not pretraind
    #total avg f1 score 0.4174473285675049 no freeze
    # total avg filtered f1 score 0.4725818634033203 no freeze
    #total avg f1 score 0.45711055397987366 freeze 
    # total avg filtered f1 score 0.4897262156009674 freeze 
    # Generate LaTeX table

    if pretrained:
        latex_table = r"""
\begin{table}[]
\begin{tabular}{|l|l|l|l|l|}
\hline
\multirow{2}{*}{\textbf{Trained on}} & \multirow{2}{*}{\textbf{Finetuned on}} & \multicolumn{3}{c|}{\textbf{Tested on}} \\ \cline{3-5}
&  & \textbf{ReFact} & \textbf{HaluEval} & \textbf{TruthfulQA} \\ \hline
"""
    else:
        latex_table = r"""
\begin{tabular}{|l|l|l|l|}
\hline
\multirow{2}{*}{\textbf{Trained on}} & \multicolumn{3}{c|}{\textbf{Tested on} & Filtered Count} \\ \cline{2-4}
                                & \textbf{ReFact} & \textbf{HaluEval} & \textbf{TruthfulQA} &\\ \hline
"""
    for latex_row in latex_rows:
        latex_table += latex_row + "\n"
    latex_table += r"""
\end{tabular}
\caption{F1 Scores for Different Benchmarks using Comparison Method: """ + comparison_mapping[comparison_method] 
    latex_table += "with freezed aggregation" if freeze else "with non-freezed aggregation"
    latex_table += "with pretrained model" if pretrained else "without pretrained model"
    latex_table += r"""}
\label{tab:in_domain_shift_f1_scores}
\end{table}
"""

    return latex_table



def load_f1_scores_and_create_table(path,final_classifier, layer_depths, comparison_methods):
    # Initialize a dictionary to store F1 scores
    f1_scores = {method: {depth: None for depth in layer_depths} for method in comparison_methods}

    # Iterate over layer depths and comparison methods to load F1 scores
    for layer_depth in layer_depths:
        for comparison_method in comparison_methods:
            file_name = f"layer_comparison_classifier_haluleval__1_{comparison_method}_{final_classifier}_{layer_depth}_False_5_f1s.pth"
            file_path = os.path.join(path, file_name)
            
            if os.path.exists(file_path):
                f1_scores[comparison_method][layer_depth] = torch.load(file_path).mean().item()
            else:
                #print(f"File not found: {file_path}")
                continue

    # Generate LaTeX table
    latex_table = r"""
\begin{table}[]
\begin{tabular}{|l|llllll|}
\hline
\multicolumn{1}{|c|}{\multirow{2}{*}{Comparison Method}} & \multicolumn{6}{c|}{Layer Depth} \\ \cline{2-7} 
\multicolumn{1}{|c|}{} & \multicolumn{1}{l|}{0} & \multicolumn{1}{l|}{1} & \multicolumn{1}{l|}{2} & \multicolumn{1}{l|}{3} & \multicolumn{1}{l|}{4} & 5 \\ \hline
"""
    for method, depths in f1_scores.items():
        latex_table += f"{method} & " + " & ".join(
            [f"{depths[depth]:.2f}" if depths[depth] is not None else "" for depth in layer_depths]
        ) + r" \\ \hline" + "\n"

    latex_table += r"""
\end{tabular}
\caption{F1 Scores for Different Comparison Methods and Layer Depths}
\label{tab:f1_scores}
\end{table}
"""
    return latex_table

def print_statistics_of_classifier(path, final_classifiers, layer_depths, comparison_methods):
    # Initialize LaTeX table
    latex_table = r"""
\begin{table}[h]
\begin{tabular}{|l|l|l|l|l|l|}
\hline
\textbf{Classifier} & \textbf{Mean} & \textbf{Median} & \textbf{Std Dev} & \textbf{Max} & \textbf{Min} \\ \hline
"""

    for final_classifier in final_classifiers:
        # Initialize a dictionary to store F1 scores
        f1_scores = {method: {depth: None for depth in layer_depths} for method in comparison_methods}
    
        # Iterate over layer depths and comparison methods to load F1 scores
        for layer_depth in layer_depths:
            for comparison_method in comparison_methods:
                file_name = f"layer_comparison_classifier_haluleval__1_{comparison_method}_{final_classifier}_{layer_depth}_False_5_f1s.pth"
                file_path = os.path.join(path, file_name)
                
                if os.path.exists(file_path):
                    f1_scores[comparison_method][layer_depth] = torch.load(file_path).mean().item()
                else:
                    continue
        
        # Create DataFrame and calculate statistics
        df = pd.DataFrame(f1_scores).dropna(axis=1)
        if not df.empty:
            mean = df.values.mean()
            median = np.median(df.values)
            std_dev = df.values.std()
            max_val = df.values.max()
            min_val = df.values.min()
        else:
            mean = median = std_dev = max_val = min_val = float('nan')

        # Add row to LaTeX table
        latex_table += f"{aggregation_mapping[final_classifier]} & {mean:.2f} & {median:.2f} & {std_dev:.2f} & {max_val:.2f} & {min_val:.2f} \\\\ \hline\n"

    latex_table += r"""
\end{tabular}
\caption{Statistics of Classifiers}
\label{tab:classifier_statistics}
\end{table}
"""
    print(latex_table)

def print_statistics_of_layers(path, final_classifiers, layer_depths, comparison_methods):
    # Initialize LaTeX table
    latex_table = r"""
\begin{table}[h]
\begin{tabular}{|l|l|l|l|l|l|l|}
\hline
\textbf{Layer Depth} & \textbf{Mean} & \textbf{Median} & \textbf{Std Dev} & \textbf{Max} & \textbf{Min} \\ \hline
"""

    # Iterate over layer depths and comparison methods to load F1 scores
    statistics = {depth: {"Mean": None, "Median": None, "Standard Deviation": None, "Max": None, "Min": None} for depth in layer_depths}
    for layer_depth in layer_depths:
        f1_scores = {final_classifier: {method: None for method in comparison_methods} for final_classifier in final_classifiers}
        for final_classifier in final_classifiers:
            for comparison_method in comparison_methods:
                file_name = f"layer_comparison_classifier_haluleval__1_{comparison_method}_{final_classifier}_{layer_depth}_False_5_f1s.pth"
                file_path = os.path.join(path, file_name)
                
                if os.path.exists(file_path):
                    f1_scores[final_classifier][comparison_method] = torch.load(file_path).mean().item()
                else:
                    continue
        df = pd.DataFrame(f1_scores).dropna(axis=1)
        if not df.empty:
            statistics[layer_depth]["Mean"] = df.values.mean()
            statistics[layer_depth]["Median"] = np.median(df.values)
            statistics[layer_depth]["Standard Deviation"] = df.values.std()
            statistics[layer_depth]["Max"] = df.values.max()
            statistics[layer_depth]["Min"] = df.values.min()

    # Add statistics to LaTeX table
    for depth, stats in statistics.items():
        latex_table += f"{depth} & {stats['Mean']:.2f} & {stats['Median']:.2f} & {stats['Standard Deviation']:.2f} & {stats['Max']:.2f} & {stats['Min']:.2f} \\\\ \hline\n"

    latex_table += r"""
\end{tabular}
\caption{Statistics for all Layer Depths combining all Model Selection Tables}
\label{tab:layer_stat}
\end{table}
"""
    print(latex_table)

def print_statistics_of_comparison_methods(path, final_classifiers, layer_depths, comparison_methods):
    # Initialize LaTeX table
    latex_table = r"""
\begin{table}[h]
\begin{tabular}{|l|lllll|}
\hline
\multicolumn{1}{|c|}{\textbf{Comparison Method}} & \textbf{Mean} & \textbf{Median} & \textbf{Std Dev} & \textbf{Max} & \textbf{Min} \\ \hline
"""

    for comparison_method in comparison_methods:
        f1_scores = {final_classifier: {depth: None for depth in layer_depths} for final_classifier in final_classifiers}
        for final_classifier in final_classifiers:
            for layer_depth in layer_depths:
                file_name = f"layer_comparison_classifier_haluleval__1_{comparison_method}_{final_classifier}_{layer_depth}_False_5_f1s.pth"
                file_path = os.path.join(path, file_name)
                
                if os.path.exists(file_path):
                    f1_scores[final_classifier][layer_depth] = torch.load(file_path).mean().item()
                else:
                    continue

        # Create DataFrame and calculate statistics
        df = pd.DataFrame(f1_scores).dropna(axis=1)
        statistics = {
            "Mean": df.values.mean(),
            "Median": np.median(df.values),
            "Standard Deviation": df.values.std(),
            "Max": df.values.max(),
            "Min": df.values.min()
        }

        # Add row to LaTeX table
        latex_table += f"{comparison_mapping[comparison_method]} & {statistics['Mean']:.2f} & {statistics['Median']:.2f} & {statistics['Standard Deviation']:.2f} & {statistics['Max']:.2f} & {statistics['Min']:.2f} \\\\ \hline\n"

    latex_table += r"""
\end{tabular}
\caption{Statistics for Comparison Methods}
\label{tab:comparison_methods_stat}
\end{table}
"""
    print(latex_table)

if __name__ == "__main__":
    path = "./thesis/data/avgs_early_stopping"
    layer_depths = [1,2,3,4,5]
    comparison_methods = [
        "no_comparison", "dot_product", "euclidean_norm", "manhatten",
        "pairwise_dot_product", "euclidean_distance", "manhatten_distance", "cosine"
    ]

    aggregation_methods=["shared_classifier_ensemble","flattend_aggregation"]
    #aggregation_methods = ["flattend_aggregation"]
    non_linearity_classifier=[True,False]

    final_layer_classifiers = [method + "_" + str(non_linearity) for method in aggregation_methods for non_linearity in non_linearity_classifier]
    """
    for final_classifier in final_layer_classifiers:
        latex_table = load_f1_scores_and_create_table(path,final_classifier=final_classifier,layer_depths=layer_depths,comparison_methods=comparison_methods)
        print(aggregation_mapping[final_classifier])
        print(latex_table)
    print_statistics_of_classifier(path,final_classifiers=final_layer_classifiers,layer_depths=layer_depths,comparison_methods=comparison_methods)


    print("Statistics of layers")
    print_statistics_of_layers(path,final_classifiers=final_layer_classifiers,layer_depths=layer_depths,comparison_methods=comparison_methods)

    print("Statistics of comparison methods")
    print_statistics_of_comparison_methods(path,final_classifiers=final_layer_classifiers,layer_depths=layer_depths,comparison_methods=comparison_methods)
    """
    print("Statistics of baselines")
    print(create_baseline_table(path, layer_depths))
    print("Statistics of LLMs")
    llm_path = "./thesis/data/different_llms_truthfulqa"
    print(create_llm_table(llm_path, comparison_method="no_comparison", benchmark_name="truthfulqa"))
    print(create_llm_table(llm_path, comparison_method="cosine", benchmark_name="truthfulqa"))
    
    


    #print("In domain shift")
    #print(create_in_domain_shift_table(comparison_method="no_comparison", pretrained=False))
    #print(create_in_domain_shift_table(comparison_method="no_comparison", pretrained=True))
    #print(create_in_domain_shift_table(comparison_method="no_comparison", pretrained=True, freeze=False))
    