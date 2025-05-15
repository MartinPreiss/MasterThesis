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
    layer_depths = [1,2]
    comparison_methods = [
        "no_comparison", "dot_product", "euclidean_norm", "manhatten",
        "pairwise_dot_product", "euclidean_distance", "manhatten_distance", "cosine"
    ]

    #aggregation_methods=["shared_classifier_ensemble","flattend_aggregation"]
    aggregation_methods = ["flattend_aggregation"]
    non_linearity_classifier=[True,False]


    final_layer_classifiers = [method + "_" + str(non_linearity) for method in aggregation_methods for non_linearity in non_linearity_classifier]
    
    for final_classifier in final_layer_classifiers:
        latex_table = load_f1_scores_and_create_table(path,final_classifier=final_classifier,layer_depths=layer_depths,comparison_methods=comparison_methods)
        print(aggregation_mapping[final_classifier])
        print(latex_table)
    print_statistics_of_classifier(path,final_classifiers=final_layer_classifiers,layer_depths=layer_depths,comparison_methods=comparison_methods)

    print("Statistics of layers")
    print_statistics_of_layers(path,final_classifiers=final_layer_classifiers,layer_depths=layer_depths,comparison_methods=comparison_methods)

    print("Statistics of comparison methods")
    print_statistics_of_comparison_methods(path,final_classifiers=final_layer_classifiers,layer_depths=layer_depths,comparison_methods=comparison_methods)