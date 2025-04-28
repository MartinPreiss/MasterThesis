import torch
import os

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


if __name__ == "__main__":
    path = "./thesis/data/avgs_early_stopping"
    layer_depths = [0, 1, 2, 3, 4, 5]
    comparison_methods = [
        "no_comparison", "dot_product", "euclidean_norm", "manhatten",
        "pairwise_dot_product", "euclidean_distance", "manhatten_distance", "cosine"
    ]

    aggregation_methods=["shared_classifier_ensemble","flattend_aggregation"]
    linearity_classifier=[True, False]


    final_layer_classifiers = [method + "_" + str(linearity) for method in aggregation_methods for linearity in linearity_classifier]
    
    for final_classifier in final_layer_classifiers:
        latex_table = load_f1_scores_and_create_table(path,final_classifier=final_classifier,layer_depths=layer_depths,comparison_methods=comparison_methods)
        print(final_classifier)
        print(latex_table)