import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_heat_map_for_tensor(tensor):
    # Create a heatmap
    sns.heatmap(tensor, annot=False, cmap='viridis')
    
    # Add titles and labels
    plt.title('Heatmap of DataFrame')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    
    # Show the plot
    plt.show()


def create_hist_for_tensor(tensors,title, xlim,ylim):

    original_values = np.concatenate([tensor.numpy().flatten() for tensor in tensors])


    # Now plot as before
    plt.figure(figsize=(10, 6))
    plt.hist(original_values, bins=30, alpha=0.7, edgecolor='black')
    plt.title(title)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    plt.clf()


def create_heat_map_for_df(df):
    # Create a heatmap
    sns.heatmap(df, annot=False, cmap='viridis')
    
    # Add titles and labels
    plt.title('Heatmap of DataFrame')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    
    # Show the plot
    plt.show()


def create_hist_for_df(df,title):
    # Concatenate all numerical values into a single array
    all_values = df.values.flatten()

    # Plot the histogram
    plt.hist(all_values, bins=20, edgecolor='black')

    # Add titles and labels
    plt.title('Histogram of '+ title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.figure(figsize=(6, 4))

    # Show the plot
    plt.show()
