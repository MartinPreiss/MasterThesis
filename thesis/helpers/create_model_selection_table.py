import torch
import os
import pandas as pd
import numpy as np

benchmark_names  = ["refact", "haluleval", "truthfulqa"]
benchmark_distributions = {
    "refact": {
        "total": 2002 * 0.15,
        "positives": 0.4849
    }, 
    "haluleval": {
        "total": 20000 * 0.15,
        "positives": 0.3273
    },
    "truthfulqa": {
        "total": 10000 * 0.15,
        "positives": 0.3659
    }
}
for benchmark_name in benchmark_names:
    num_positives = benchmark_distributions[benchmark_name]["positives"] * benchmark_distributions[benchmark_name]["total"]
    num_negatives = benchmark_distributions[benchmark_name]["total"] - num_positives
    print(f"Benchmark: {benchmark_name}, Positives: {num_positives}, Negatives: {num_negatives}")
    print("accuracy = ", num_negatives / (num_negatives + num_positives))
    benchmark_distributions[benchmark_name]["accuracy"] = num_negatives / (num_negatives + num_positives)

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
    f1_scores = []
    total_filtered_count = 0
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

            f1_scores.append(test_f1_scores)
            total_filtered_count += filtered_count

            latex_rows.append(f"{llm} & {original_mean:.2f} & {filtered_mean:.2f} & {filtered_count} \\\\ \\hline") 
        else:
            #print(f"File not found: {file_path}")
            latex_rows.append(f"{llm} & N/A & N/A & N/A \\\\ \\hline")

            continue
    #calculate some statistics 
    total_avg = torch.cat(f1_scores).mean().item()
    without_devllm_avg = torch.cat(f1_scores[1:]).mean().item() # without devllm


    print(f"Total Average F1 Score: {total_avg:.2f}")
    print(f"Total Average F1 Score without devllm: {without_devllm_avg:.2f}")
    print(f"Total Filtered Count: {total_filtered_count}")

            
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

def create_in_domain_shift_table(comparison_method="", pretrained=True, freeze=True, model_name="", path=None):
    # thesis/data/avg_in_domain_shift/haluleval_refact_results.pth
    if path is None:
        if pretrained:
            if freeze:
                path = "./thesis/data/avg_in_domain_shift_pretrained_original_hyp"
            else:
                path = "./thesis/data/avg_in_domain_shift_pretrained_no_freeze"
        else:
            path =  "./thesis/data/avg_in_domain_shift_original_hyp"
    
    filename_splitter = "__" if comparison_method == "" else "_"
    files = os.listdir(path) 
    latex_rows = []
    all_test_scores = []
    benchmark_names  = ["refact", "haluleval", "truthfulqa"]
    for file in files: 
        if not comparison_method in file: 
            continue 
        if not model_name in file:
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
        if model_name is not "":
            file = file.split(filename_splitter,maxsplit=1)[1]
        if pretrained: 
            first_benchmark_name = file.split(filename_splitter)[0]
            second_benchmark_name = file.split(filename_splitter)[1]

            results_concatenated = " & ".join([f"{test_result[1]:.2f}" for test_result in benchmark_results])
            latex_row = f"{first_benchmark_name} & {second_benchmark_name} & {results_concatenated} & {filtered_count} \\\\ \\hline"

        else:
            second_benchmark_name = file.split("__")[1]
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
\caption{F1 Scores for Different Benchmarks using Comparison Method: """
    latex_table += comparison_mapping[comparison_method] if comparison_method is not "" else ""
    latex_table += "with freezed aggregation" if freeze else "with non-freezed aggregation"
    latex_table += "with pretrained model" if pretrained else "without pretrained model"
    latex_table += "with " + model_name if model_name != "" else ""
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

def create_positinoal_latex_table(with_crf=False):
    path = "thesis/data/positions"
    tagging_schemes = ["IO", "BIO", "BIOES"]
    comparison_method = ["no_comparison", "cosine"]
    patience = 5 
    num_runs = 1 
    #example_filename = thesis/data/positions/layer_comparison_classifier__BIO__cosine__5__1
    #example_filenMe = thesis/data/positions/lcc_with_crf__BIO__cosine__5__1

    if with_crf:
        model_name = "lcc_with_crf"
    else:
        model_name = "layer_comparison_classifier"
    all_f1_scores = []
    latex_rows = []
    for comparison in comparison_method:   
        latex_row = f"{comparison_mapping[comparison]} & "
        filtered_f1_scores = []
        for tagging_scheme in tagging_schemes:
            file_name = f"{model_name}__{tagging_scheme}__{comparison}__{patience}__{num_runs}.pth"
            file_path = os.path.join(path, file_name)
            if os.path.exists(file_path):
                result_dict = torch.load(file_path, weights_only=False)
                test_results = result_dict["test_results"]
                val_results = result_dict["val_results"]
                #postprocess the f1 score dict
                val_f1_scores = torch.tensor([val_score_dict["f1"] for val_score_dict in val_results])
                test_f1_scores = torch.tensor([test_score_dict["f1"] for test_score_dict in test_results])
                #filter out f1 scores that are zero
                original_mean = test_f1_scores.mean().item()
                filtered_count = val_f1_scores[val_f1_scores == 0].shape[0]
                test_f1_scores = test_f1_scores[val_f1_scores != 0]
                filtered_mean = test_f1_scores.mean().item()
                filtered_f1_scores.append(original_mean)
        latex_row += "&".join([f"{score:.2f}" for score in filtered_f1_scores])
        all_f1_scores.append(torch.tensor(filtered_f1_scores))
        latex_rows.append(latex_row + f" \\\\ \\hline")

    
    #baselines 
    baselines = ["last_layer", "middle_layer", "stacked_layers", "all_layers_ensemble"]
    if with_crf: 
        baselines = ["baseline_with_crf_"+ baseline for baseline in baselines]
    comparison_method = "None"
    for baseline in baselines:
        latex_row = f"{baseline} & "
        filtered_f1_scores = []
        for tagging_scheme in tagging_schemes:
            file_name = f"{baseline}__{tagging_scheme}__{comparison_method}__{patience}__{num_runs}.pth"
            file_path = os.path.join(path, file_name)
            if os.path.exists(file_path):
                result_dict = torch.load(file_path, weights_only=False)
                test_results = result_dict["test_results"]
                val_results = result_dict["val_results"]
                #postprocess the f1 score dict
                val_f1_scores = torch.tensor([val_score_dict["f1"] for val_score_dict in val_results])
                test_f1_scores = torch.tensor([test_score_dict["f1"] for test_score_dict in test_results])
                #filter out f1 scores that are zero
                original_mean = test_f1_scores.mean().item()
                filtered_count = val_f1_scores[val_f1_scores == 0].shape[0]
                test_f1_scores = test_f1_scores[val_f1_scores != 0]
                filtered_mean = test_f1_scores.mean().item()
                filtered_f1_scores.append(original_mean)
        latex_row += "&".join([f"{score:.2f}" for score in filtered_f1_scores])
        all_f1_scores.append(torch.tensor(filtered_f1_scores))
        latex_rows.append(latex_row + f" \\\\ \\hline")

                
    #calculate some statistics
    all_f1_scores = torch.stack(all_f1_scores)
    print(all_f1_scores)
    mean_f1_scores = all_f1_scores.mean().item()
    print("Mean F1 Scores:", mean_f1_scores)

    #first two rows are the comparison methods, last four rows are the baselines
    method_mean = all_f1_scores[:2].mean().tolist()
    baseline_mean = all_f1_scores[2:].mean().tolist()
    print("Mean F1 Scores for Methods:", method_mean)
    print("Mean F1 Scores for Baselines:", baseline_mean)

    # every column is a tagging scheme 
    io_mean = all_f1_scores[:, 0].mean().item()
    bio_mean = all_f1_scores[:, 1].mean().item()
    bioes_mean = all_f1_scores[:, 2].mean().item()
    print("Mean F1 Scores for IO:", io_mean)
    print("Mean F1 Scores for BIO:", bio_mean)
    print("Mean F1 Scores for BIOES:", bioes_mean)

    latext_table = r"""\begin{table}[h]
\begin{tabular}{|l|l|l|l|}
\hline
\multirow{2}{*}{\textbf{Model}} & \multicolumn{3}{c|}{\textbf{Tagging Scheme}} \\ \cline{2-4}
                                & \textbf{IO} & \textbf{BIO} & \textbf{BIOES} \\ \hline
"""
    for latex_row in latex_rows:
        latext_table += latex_row + "\n"
    latext_table += r"""
\end{tabular}
\caption{F1 Scores for the Localization Experiment}
\label{tab:f1_scores}
\end{table}
"""
    return latext_table

def create_other_positinal_metrics_latex_table(path, with_crf=False):

    # columns model, tagging scheme, accuracy, precision, recall, f1 score
    tagging_schemes = ["IO", "BIO", "BIOES"]
    comparison_method = ["no_comparison", "cosine"]
    patience = 5
    num_runs = 1
    if with_crf:
        model_name = "lcc_with_crf"
    else:
        model_name = "layer_comparison_classifier"
    latex_rows = []
    for comparison in comparison_method:
        for tagging_scheme in tagging_schemes:
            latex_row = f"{comparison_mapping[comparison]} & {tagging_scheme} & "
            file_name = f"{model_name}__{tagging_scheme}__{comparison}__{patience}__{num_runs}.pth"
            file_path = os.path.join(path, file_name)
            if os.path.exists(file_path):
                test_results = torch.load(file_path, weights_only=False)["test_results"]
                #postprocess the f1 score dict
                accuracy = torch.tensor([test_score_dict["acc"] for test_score_dict in test_results]).mean().item()
                precision = torch.tensor([test_score_dict["prec"] for test_score_dict in test_results]).mean().item()
                recall = torch.tensor([test_score_dict["rec"] for test_score_dict in test_results]).mean().item()
                f1_score = torch.tensor([test_score_dict["f1"] for test_score_dict in test_results]).mean().item()
                latex_row += f"{accuracy:.2f} & {precision:.2f} & {recall:.2f} & {f1_score:.2f}"
            else:
                latex_row += "N/A & N/A & N/A & N/A & N/A"
            latex_rows.append(latex_row + f" \\\\ \\hline")
    # baselines
    baselines = ["last_layer", "middle_layer", "stacked_layers", "all_layers_ensemble"]
    if with_crf:
        baselines = ["baseline_with_crf_" + baseline for baseline in baselines]

    for baseline in baselines:
        for tagging_scheme in tagging_schemes:
            latex_row = f"{baseline} & {tagging_scheme} & "
            comparison_method = "None"
            file_name = f"{baseline}__{tagging_scheme}__{comparison_method}__{patience}__{num_runs}.pth"
            file_path = os.path.join(path, file_name)
            if os.path.exists(file_path):
                test_results = torch.load(file_path, weights_only=False)["test_results"]
                #postprocess the f1 score dict
                accuracy = torch.tensor([test_score_dict["acc"] for test_score_dict in test_results]).mean().item()
                precision = torch.tensor([test_score_dict["prec"] for test_score_dict in test_results]).mean().item()
                recall = torch.tensor([test_score_dict["rec"] for test_score_dict in test_results]).mean().item()
                f1_score = torch.tensor([test_score_dict["f1"] for test_score_dict in test_results]).mean().item()
                latex_row += f"{accuracy:.2f} & {precision:.2f} & {recall:.2f} & {f1_score:.2f}"
            else:
                latex_row += "N/A & N/A & N/A & N/A & N/A"
            latex_rows.append(latex_row + f" \\\\ \\hline")
    latext_table = r"""\begin{table}[h]
\begin{tabular}{|l|l|l|l|l|l|l|}
\hline
Model & Tagging Scheme & Accuracy & Precision & Recall & F1 Score \\ \hline
"""

    for latex_row in latex_rows:
        latext_table += latex_row + "\n"
    latext_table += r"""
\end{tabular}
\caption{Other Metrics for the Localization Experiment}we
\label{tab:other_metrics}
\end{table}
"""
    return latext_table

def print_statistics_of_indomain_shift_tables():

    baseline_paths = ["thesis/data/avg_in_domain_shift"]#, "thesis/data/avg_in_domain_shift_pretrained"]
    #method_paths = ["./thesis/data/avg_in_domain_shift_original_hyp","./thesis/data/avg_in_domain_shift_pretrained_no_freeze"]# ]#,"./thesis/data/avg_in_domain_shift_pretrained_original_hyp" ]

    method_paths = ["./thesis/data/avg_in_domain_shift_original_hyp"]
    #method_paths = ["./thesis/data/avg_in_domain_shift_pretrained_no_freeze"]
    #method_paths = ["./thesis/data/avg_in_domain_shift_pretrained_original_hyp"]
    benchmark_names = ["refact", "haluleval", "truthfulqa"]
    # collect f1 scores of baselines and then methods 
    baseline_f1_scores =[]
    baseline_filtered_f1_scores = []
    baseline_filtered_count = 0
    total_baseline_runs = 0
    baseline_domain_shift_f1_scores = []
    baseline_domain_shift_filtered_f1_scores = []
    baseline_lastly_trained = []
    
    #baseline_to_filter = "last_layer" # or "middle_layer", "stacked_layers", "all_layers_ensemble"
    for path in baseline_paths:
        for file in os.listdir(path):
            #if not (baseline_to_filter in file):
            #    print(file)
            #    continue
            result_dict = torch.load(os.path.join(path, file))
            test_results = result_dict["test_results"]
            
            val_results = result_dict["val_results"]
            val_f1_scores = torch.tensor([val_score_dict["f1"] for val_score_dict in val_results])
            total_baseline_runs += val_f1_scores.shape[0]
            baseline_filtered_count += val_f1_scores[val_f1_scores == 0].shape[0]
            
            file = file.split("__",maxsplit=1)[1]

            #if "pretrained" in path:
            #    not_seen_benchmarks = [benchmark_name for benchmark_name in benchmark_names if not benchmark_name in file]
            #else: 
            trained_on_benchmark = file.split("__")[1]
            not_seen_benchmarks = [benchmark_name for benchmark_name in benchmark_names if not benchmark_name in trained_on_benchmark]
            print(file)
            print("trained on benchmark", trained_on_benchmark)
            print("not seen benchmarks", not_seen_benchmarks)

            for benchmark_name in benchmark_names:
                f1_scores = torch.tensor([test_score_dict[benchmark_name]["f1"] for test_score_dict in test_results])
                baseline_f1_scores.append(f1_scores)
                #filter out f1 scores that are zero
                filtered_f1_scores = f1_scores[val_f1_scores != 0]
                baseline_filtered_f1_scores.append(filtered_f1_scores)
                print(file)
                print(not_seen_benchmarks)
                if benchmark_name in not_seen_benchmarks:
                    baseline_domain_shift_f1_scores.append(f1_scores)
                    baseline_domain_shift_filtered_f1_scores.append(filtered_f1_scores)
                else: 
                    baseline_lastly_trained.append(filtered_f1_scores)
                
            
    baseline_f1_scores = torch.cat(baseline_f1_scores)
    baseline_filtered_f1_scores = torch.cat(baseline_filtered_f1_scores)
    baseline_domain_shift_f1_scores = torch.cat(baseline_domain_shift_f1_scores)
    baseline_domain_shift_filtered_f1_scores = torch.cat(baseline_domain_shift_filtered_f1_scores)
    baseline_lastly_trained = torch.cat(baseline_lastly_trained)
    
    method_f1_scores = []
    method_filtered_f1_scores = []
    method_filtered_count = 0
    total_method_runs = 0
    method_domain_shift_f1_scores = []
    method_domain_shift_filtered_f1_scores = []
    method_lastly_trained_filtered = []
    filter_for_method = "cosine" # or "no_comparison"
    for path in method_paths:
        for file in os.listdir(path):
            if not (filter_for_method in file):
                print(file)
                continue
            result_dict = torch.load(os.path.join(path, file))
            test_results = result_dict["test_results"]
            val_results = result_dict["val_results"]

            val_f1_scores = torch.tensor([val_score_dict["f1"] for val_score_dict in val_results])
            total_method_runs += val_f1_scores.shape[0]
            method_filtered_count += val_f1_scores[val_f1_scores == 0].shape[0]

            #first_benchmark = file.split("_")[0]
            #if not( first_benchmark == "haluleval"):
            #    print("first_benchmark",first_benchmark)
            #    continue
            file = file.split("_",maxsplit=1)[1] 
            trained_on_benchmark = file.split("_")[0]
            
            #if "pretrained" in path:
            #    not_seen_benchmarks = [benchmark_name for benchmark_name in benchmark_names if not benchmark_name in file]
            #else:
            not_seen_benchmarks = [benchmark_name for benchmark_name in benchmark_names if not benchmark_name in trained_on_benchmark]

            for benchmark_name in benchmark_names:
                f1_scores = torch.tensor([test_score_dict[benchmark_name]["f1"] for test_score_dict in test_results])
                method_f1_scores.append(f1_scores)
                #filter out f1 scores that are zero
                filtered_f1_scores = f1_scores[val_f1_scores != 0]
                method_filtered_f1_scores.append(filtered_f1_scores)
                assert trained_on_benchmark not in not_seen_benchmarks, f"trained on benchmark {trained_on_benchmark} is in not seen benchmarks {not_seen_benchmarks}"
                if benchmark_name in not_seen_benchmarks:
                    method_domain_shift_f1_scores.append(f1_scores)
                    method_domain_shift_filtered_f1_scores.append(filtered_f1_scores)
                if benchmark_name in trained_on_benchmark:
                    method_lastly_trained_filtered.append(filtered_f1_scores)

    method_f1_scores = torch.cat(method_f1_scores)
    method_filtered_f1_scores = torch.cat(method_filtered_f1_scores)
    method_domain_shift_f1_scores = torch.cat(method_domain_shift_f1_scores)
    method_domain_shift_filtered_f1_scores = torch.cat(method_domain_shift_filtered_f1_scores)

    # normalize count 
    baseline_filtered_count = baseline_filtered_count / total_baseline_runs
    method_filtered_count = method_filtered_count / total_method_runs
    # calculate statistics
    baseline_mean = baseline_f1_scores.mean().item()
    baseline_std = baseline_f1_scores.std().item()

    method_mean = method_f1_scores.mean().item()
    method_std = method_f1_scores.std().item()

    # print statistics
    print(f"Baseline Mean: {baseline_mean:.2f}, Baseline Std: {baseline_std:.2f}")
    print(f"Method Mean: {method_mean:.2f}, Method Std: {method_std:.2f}")


    # print filtered statistics
    baseline_filtered_mean = baseline_filtered_f1_scores.mean().item()
    baseline_filtered_std = baseline_filtered_f1_scores.std().item()
    method_filtered_mean = method_filtered_f1_scores.mean().item()
    method_filtered_std = method_filtered_f1_scores.std().item()
    print(f"Baseline Filtered Mean: {baseline_filtered_mean:.2f}, Baseline Filtered Std: {baseline_filtered_std:.2f}")
    print(f"Method Filtered Mean: {method_filtered_mean:.2f}, Method Filtered Std: {method_filtered_std:.2f}")
    print(f"Baseline Filtered Count: {baseline_filtered_count}")
    print(f"Method Filtered Count: {method_filtered_count}")

    # print domain shift statistics
    baseline_domain_shift_mean = baseline_domain_shift_f1_scores.mean().item()
    baseline_domain_shift_std = baseline_domain_shift_f1_scores.std().item()
    method_domain_shift_mean = method_domain_shift_f1_scores.mean().item()
    method_domain_shift_std = method_domain_shift_f1_scores.std().item()
    print(f"Baseline Domain Shift Mean: {baseline_domain_shift_mean:.2f}, Baseline Domain Shift Std: {baseline_domain_shift_std:.2f}")
    print(f"Method Domain Shift Mean: {method_domain_shift_mean:.2f}, Method Domain Shift Std: {method_domain_shift_std:.2f}")

    # print filtered domain shift statistics
    baseline_domain_shift_filtered_mean = baseline_domain_shift_filtered_f1_scores.mean().item()
    baseline_domain_shift_filtered_std = baseline_domain_shift_filtered_f1_scores.std().item()
    method_domain_shift_filtered_mean = method_domain_shift_filtered_f1_scores.mean().item()
    method_domain_shift_filtered_std = method_domain_shift_filtered_f1_scores.std().item()
    print(f"Baseline Domain Shift Filtered Mean: {baseline_domain_shift_filtered_mean:.2f}, Baseline Domain Shift Filtered Std: {baseline_domain_shift_filtered_std:.2f}")
    print(f"Method Domain Shift Filtered Mean: {method_domain_shift_filtered_mean:.2f}, Method Domain Shift Filtered Std: {method_domain_shift_filtered_std:.2f}")


    #print lastly trained filtered statistics
    method_lastly_trained_filtered_mean = torch.cat(method_lastly_trained_filtered).mean().item()
    method_lastly_trained_filtered_std = torch.cat(method_lastly_trained_filtered).std().item()
    baseline_lastly_trained_filtered_mean = baseline_lastly_trained.mean().item()
    baseline_lastly_trained_filtered_std = baseline_lastly_trained.std().item()
    print(f"Baseline Lastly Trained Filtered Mean: {baseline_lastly_trained_filtered_mean:.2f}, Baseline Lastly Trained Filtered Std: {baseline_lastly_trained_filtered_std:.2f}")
    print(f"Method Lastly Trained Filtered Mean: {method_lastly_trained_filtered_mean:.2f}, Method Lastly Trained Filtered Std: {method_lastly_trained_filtered_std:.2f}")

def calculate_ptrue_improvement(comparison_method="", pretrained=True, freeze=True, model_name="", path=None): 
    # indomain shift resultscomparison_method="", pretrained=True, freeze=True, model_name="", path=None):
    # thesis/data/avg_in_domain_shift/haluleval_refact_results.pth
    if path is None:
        if pretrained:
            if freeze:
                path = "./thesis/data/avg_in_domain_shift_pretrained_original_hyp"
            else:
                path = "./thesis/data/avg_in_domain_shift_pretrained_no_freeze"
        else:
            path =  "./thesis/data/avg_in_domain_shift_original_hyp"
    
    filename_splitter = "__" if comparison_method == "" else "_"
    files = os.listdir(path) 
    latex_rows = []
    

    for file in files: 
        if not comparison_method in file: 
            continue 
        if not model_name in file:
            continue
        result_dict = torch.load(os.path.join(path, file))
        test_results = result_dict["test_results"]
        val_results = result_dict["val_results"]
        #postprocess the f1 score dict 
        val_f1_scores = torch.tensor([val_score_dict["f1"] for val_score_dict in val_results])
    

        filtered_count = val_f1_scores[val_f1_scores == 0].shape[0]
        benchmark_results = []
        for benchmark_name in benchmark_names:
            test_dicts = [test_score_dict[benchmark_name] for test_score_dict in test_results]
            pos_ratio = benchmark_distributions[benchmark_name]["positives"]
            total_samples_count = benchmark_distributions[benchmark_name]["total"]
            num_positives = pos_ratio * total_samples_count
            num_negatives = total_samples_count - num_positives
            
            accs = []
            for test_dict in test_dicts:
                recall = test_dict["rec"]
                precision = test_dict["prec"]
                if precision == 0 or recall == 0:
                    accs.append(0.0)
                    continue

                tps = recall * num_positives
                fps = (tps / precision) - tps

                new_num_positives = num_positives - tps
                new_num_negatives = num_negatives + tps
                new_num_positives +=  fps
                new_num_negatives -= fps
                new_accuracy = new_num_negatives / (new_num_negatives + new_num_positives)
                accuracy_change = new_accuracy - benchmark_distributions[benchmark_name]["accuracy"]
                accs.append(accuracy_change*100)
            accs = torch.tensor(accs)
            #filter with val f1 
            accs = accs[val_f1_scores != 0]
            avg_acc = accs.mean().item()
            benchmark_results.append(avg_acc)
        if model_name is not "":
            file = file.split(filename_splitter,maxsplit=1)[1]
        if pretrained: 
            first_benchmark_name = file.split(filename_splitter)[0]
            second_benchmark_name = file.split(filename_splitter)[1]

            results_concatenated = " & ".join([f"{test_result:.2f}" for test_result in benchmark_results])
            latex_row = f"{first_benchmark_name} & {second_benchmark_name} & {results_concatenated} & {filtered_count} \\\\ \\hline"

        else:
            second_benchmark_name = file.split(filename_splitter)[1]
            benchmark_results = [f"{test_result:.2f}" for test_result in benchmark_results]
            latex_row = f"{second_benchmark_name} & " + " & ".join(benchmark_results) + f" & {filtered_count} \\\\ \\hline"
        latex_rows.append(latex_row)

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
\caption{F1 Scores for Different Benchmarks using Comparison Method: """
    latex_table += comparison_mapping[comparison_method] if comparison_method is not "" else ""
    latex_table += "with freezed aggregation" if freeze else "with non-freezed aggregation"
    latex_table += "with pretrained model" if pretrained else "without pretrained model"
    latex_table += "with " + model_name if model_name != "" else ""
    latex_table += r"""}
\label{tab:in_domain_shift_f1_scores}
\end{table}
"""

    return latex_table

def create_ptrue_statistic():
    methods_path = "./thesis/data/avg_in_domain_shift_original_hyp"
    methods_path = "./thesis/data/avg_in_domain_shift_pretrained_no_freeze"
    methods_path = "./thesis/data/avg_in_domain_shift_pretrained_original_hyp"
    

    baseline_path = "thesis/data/avg_in_domain_shift"

    #like indomain shift 
    #compare how much improvement on lastly_trained and indomain shits happened 
    method_domain_shift_ptrue_change = []
    method_lastly_trained_ptrue_change = []
    
    #method_to_filter = "cosine" # or "no_comparison"
    for file in os.listdir(methods_path):
        #if not (method_to_filter in file):
        #    print(file)	
        #    continue
        result_dict = torch.load(os.path.join(methods_path, file))
        test_results = result_dict["test_results"]
        val_results = result_dict["val_results"]
        #postprocess the f1 score dict 
        val_f1_scores = torch.tensor([val_score_dict["f1"] for val_score_dict in val_results])

        trained_on_benchmark = file.split("_")[1]
        not_seen_benchmarks = [benchmark_name for benchmark_name in benchmark_names if not benchmark_name in trained_on_benchmark]
        for benchmark_name in benchmark_names:
            test_dicts = [test_score_dict[benchmark_name] for test_score_dict in test_results]
            pos_ratio = benchmark_distributions[benchmark_name]["positives"]
            total_samples_count = benchmark_distributions[benchmark_name]["total"]
            num_positives = pos_ratio * total_samples_count
            num_negatives = total_samples_count - num_positives
            
            accs = []
            for test_dict in test_dicts:
                recall = test_dict["rec"]
                precision = test_dict["prec"]
                if precision == 0 or recall == 0:
                    accs.append(0.0)
                    continue

                tps = recall * num_positives
                fps = (tps / precision) - tps

                new_num_positives = num_positives - tps
                new_num_negatives = num_negatives + tps
                new_num_positives +=  fps
                new_num_negatives -= fps
                new_accuracy = new_num_negatives / (new_num_negatives + new_num_positives)
                accuracy_change = new_accuracy - benchmark_distributions[benchmark_name]["accuracy"]
                accs.append(accuracy_change*100)
            accs = torch.tensor(accs)

            #filter with val f1 
            accs = accs[val_f1_scores != 0]
            if benchmark_name in not_seen_benchmarks:
                method_domain_shift_ptrue_change.append(accs)
            if benchmark_name in trained_on_benchmark:
                method_lastly_trained_ptrue_change.append(accs)
    method_domain_shift_ptrue_change = torch.cat(method_domain_shift_ptrue_change)
    method_lastly_trained_ptrue_change = torch.cat(method_lastly_trained_ptrue_change)
    method_all_ptrue_change = torch.cat([method_domain_shift_ptrue_change, method_lastly_trained_ptrue_change])
    method_domain_shift_mean = method_domain_shift_ptrue_change.mean().item()
    method_domain_shift_std = method_domain_shift_ptrue_change.std().item()
    method_lastly_trained_mean = method_lastly_trained_ptrue_change.mean().item()
    method_lastly_trained_std = method_lastly_trained_ptrue_change.std().item()
    method_all_mean = method_all_ptrue_change.mean().item()
    method_all_std = method_all_ptrue_change.std().item()
    print(f"Method All Mean: {method_all_mean:.2f}, Method All Std: {method_all_std:.2f}")
    print(f"Method Domain Shift Mean: {method_domain_shift_mean:.2f}, Method Domain Shift Std: {method_domain_shift_std:.2f}")
    print(f"Method Lastly Trained Mean: {method_lastly_trained_mean:.2f}, Method Lastly Trained Std: {method_lastly_trained_std:.2f}")

    #baseline
    baseline_domain_shift_ptrue_change = []
    baseline_lastly_trained_ptrue_change = []
    baseline_to_filter = "stacked_layers" # or "middle_layer", "stacked_layers", "all_layers_ensemble"
    for file in os.listdir(baseline_path):
        if not baseline_to_filter in file:
            continue
        result_dict = torch.load(os.path.join(baseline_path, file))
        test_results = result_dict["test_results"]
        val_results = result_dict["val_results"]
        #postprocess the f1 score dict 
        val_f1_scores = torch.tensor([val_score_dict["f1"] for val_score_dict in val_results])

        file = file.split("__",maxsplit=1)[1]
        trained_on_benchmark = file.split("__")[1]
        not_seen_benchmarks = [benchmark_name for benchmark_name in benchmark_names if not benchmark_name in trained_on_benchmark]
        for benchmark_name in benchmark_names:
            test_dicts = [test_score_dict[benchmark_name] for test_score_dict in test_results]
            pos_ratio = benchmark_distributions[benchmark_name]["positives"]
            total_samples_count = benchmark_distributions[benchmark_name]["total"]
            num_positives = pos_ratio * total_samples_count
            num_negatives = total_samples_count - num_positives
            
            accs = []
            for test_dict in test_dicts:
                recall = test_dict["rec"]
                precision = test_dict["prec"]
                if precision == 0 or recall == 0:
                    accs.append(0.0)
                    continue

                tps = recall * num_positives
                fps = (tps / precision) - tps

                new_num_positives = num_positives - tps
                new_num_negatives = num_negatives + tps
                new_num_positives +=  fps
                new_num_negatives -= fps
                new_accuracy = new_num_negatives / (new_num_negatives + new_num_positives)
                accuracy_change = new_accuracy - benchmark_distributions[benchmark_name]["accuracy"]
                accs.append(accuracy_change*100)
            accs = torch.tensor(accs)

            #filter with val f1 
            accs = accs[val_f1_scores != 0]
            if benchmark_name in not_seen_benchmarks:
                baseline_domain_shift_ptrue_change.append(accs)
            if benchmark_name in trained_on_benchmark:
                baseline_lastly_trained_ptrue_change.append(accs)
    baseline_domain_shift_ptrue_change = torch.cat(baseline_domain_shift_ptrue_change)
    baseline_lastly_trained_ptrue_change = torch.cat(baseline_lastly_trained_ptrue_change)
    baseline_total_mean = torch.cat([baseline_domain_shift_ptrue_change, baseline_lastly_trained_ptrue_change]).mean().item()
    baseline_domain_shift_mean = baseline_domain_shift_ptrue_change.mean().item()
    baseline_domain_shift_std = baseline_domain_shift_ptrue_change.std().item()
    baseline_lastly_trained_mean = baseline_lastly_trained_ptrue_change.mean().item()
    baseline_lastly_trained_std = baseline_lastly_trained_ptrue_change.std().item()
    print(f"Baseline Total Mean: {baseline_total_mean:.2f}")
    print(f"Baseline Domain Shift Mean: {baseline_domain_shift_mean:.2f}, Baseline Domain Shift Std: {baseline_domain_shift_std:.2f}")
    print(f"Baseline Lastly Trained Mean: {baseline_lastly_trained_mean:.2f}, Baseline Lastly Trained Std: {baseline_lastly_trained_std:.2f}")
def create_llm_statistic(): 
    path =  "./thesis/data/different_llms_truthfulqa"
    benchmark_name = "truthfulqa"
    comparison_methods = ["no_comparison", "cosine"]
    #comparison_methods = ["no_comparison"]
    #comparison_methods = ["cosine"]
    gemma_f1_scores = []
    llama_f1_scores = []
    llms = ["meta-llama/Llama-3.1-8B-Instruct" ,"google/gemma-3-1b-it" ,"google/gemma-3-4b-it" ,"google/gemma-3-12b-it" ,"google/gemma-3-27b-it" ,"meta-llama/Llama-3.2-1B-Instruct" ,"meta-llama/Llama-3.2-3B-Instruct" ,"meta-llama/Llama-3.3-70B-Instruct"]
    llms = [llm[llm.rfind("/")+1:] for llm in llms]
    for llm in llms:
        for comparison_method in comparison_methods:
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
                if "gemma" in llm.lower():
                    gemma_f1_scores.append(test_f1_scores)
                elif "llama" in llm.lower():
                    llama_f1_scores.append(test_f1_scores)
                else:
                    print(f"Unknown LLM: {llm}")
    #get avg for gemma and llama
    gemma_f1_scores = torch.cat(gemma_f1_scores)
    llama_f1_scores = torch.cat(llama_f1_scores)
    gemma_mean = gemma_f1_scores.mean().item()
    gemma_std = gemma_f1_scores.std().item()
    llama_mean = llama_f1_scores.mean().item()
    llama_std = llama_f1_scores.std().item()
    print(f"Gemma Mean: {gemma_mean:.2f}, Gemma Std: {gemma_std:.2f}")
    print(f"Llama Mean: {llama_mean:.2f}, Llama Std: {llama_std:.2f}")



if __name__ == "__main__":
    """
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
    """
    #print("Statistics of baselines")
    #print(create_baseline_table(path, layer_depths))
    #print("Statistics of LLMs")
    #llm_path = "./thesis/data/different_llms_truthfulqa"
    #print(create_llm_table(llm_path, comparison_method="no_comparison", benchmark_name="truthfulqa"))
    #print(create_llm_table(llm_path, comparison_method="cosine", benchmark_name="truthfulqa"))
    
    #print("In domain shift")
    #print(create_in_domain_shift_table(comparison_method="no_comparison", pretrained=False))
    #print(create_in_domain_shift_table(comparison_method="cosine", pretrained=True))
    #print(create_in_domain_shift_table(comparison_method="cosine", pretrained=True, freeze=False))
    #print(create_in_domain_shift_table(comparison_method="no_comparison", pretrained=True))
    #print(create_in_domain_shift_table(comparison_method="no_comparison", pretrained=True, freeze=False))
    #
    #"indomain shift baselines"
    # baselines = ["last_layer", "middle_layer", "stacked_layers", "all_layers_ensemble"]
    # for baseline in baselines:
    #     print(create_in_domain_shift_table(path="thesis/data/avg_in_domain_shift",model_name=baseline,pretrained=False))
    #     print(create_in_domain_shift_table(path="thesis/data/avg_in_domain_shift_pretrained",model_name=baseline))
    # print("Positional dataset")
    #print(create_positinoal_latex_table(with_crf=True))
    #print("Statistics of in domain shift tables")
    #print_statistics_of_indomain_shift_tables()

    

    #print(create_other_positinal_metrics_latex_table(path="thesis/data/positions", with_crf=True))
    #print("Calculate ptrue improvement")
    #print(calculate_ptrue_improvement(comparison_method="cosine", pretrained=True, freeze=True))
    #print(calculate_ptrue_improvement(comparison_method="no_comparison", pretrained=True, freeze=True))
    #baselines = ["last_layer", "middle_layer", "stacked_layers", "all_layers_ensemble"]
    #for baseline in baselines:
    #     print(calculate_ptrue_improvement(path="thesis/data/avg_in_domain_shift",model_name=baseline,pretrained=False))


    #print(create_llm_statistic())

    create_ptrue_statistic()