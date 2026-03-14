import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import numpy as np
from collections import defaultdict

def load_and_filter_data(file_config_list):
    """
    Reads multiple JSON files from various directories and aggregates them.
    """
    combined_results = []
    search_dirs = ['', 'Outputs', 'Outputs_t2', 'Outputs_t3', 'Outputs_t4', 'Outputs_t5']

    for filename, file_threshold in file_config_list:
        file_found = False
        path_to_open = None

        for folder in search_dirs:
            possible_path = os.path.join(folder, filename) if folder else filename
            if os.path.exists(possible_path):
                path_to_open = possible_path
                file_found = True
                break

        if not file_found:
            print(f"Warning: Could not find file '{filename}'. Skipping.")
            continue

        try:
            with open(path_to_open, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    filtered_data = [d for d in data if d.get('best_val_corr', -1) >= file_threshold]
                    combined_results.extend(filtered_data)
                    print(f"Loaded {path_to_open}: kept {len(filtered_data)}/{len(data)}")
        except Exception as e:
            print(f"Error reading {path_to_open}: {e}")
            
    return combined_results

def organize_data(results):
    """
    Helper function to organize raw results into a dictionary:
    Key: (hidden_dim, dropout)
    Value: list of (learning_rate, correlation) sorted by LR
    """
    grouped_data = defaultdict(list)
    
    for exp in results:
        config = exp['config']
        h_dim = config.get('hidden_dim', 'N/A')
        drop = config.get('dropout', 'N/A')
        lr = config.get('learning_rate')
        corr = exp.get('best_val_corr')
        
        if lr is not None and corr is not None:
            key = (h_dim, drop)
            grouped_data[key].append((lr, corr))
            
    # Sort points within each group by LR for clean plotting
    for key in grouped_data:
        grouped_data[key].sort(key=lambda x: x[0])
        
    return grouped_data

def plot_individual_configs(grouped_data, model_name, output_dir):
    """
    Generates one PDF per configuration (Isolated View).
    """
    print(f"   -> Generating individual plots...")
    for (h_dim, drop), points in grouped_data.items():
        lrs = [p[0] for p in points]
        corrs = [p[1] for p in points]

        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(lrs, corrs, color='navy', linestyle='--', alpha=0.5, zorder=2)
        ax.scatter(lrs, corrs, color='crimson', s=100, zorder=3, edgecolors='black')

        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate (Log Scale)', fontsize=12)
        ax.set_ylabel('Validation Spearman Correlation', fontsize=12)
        ax.set_title(f"{model_name} (Isolated): H={h_dim}, Drop={drop}", fontsize=14)
        ax.grid(True, which="both", ls="--", alpha=0.4)

        filename = f"{model_name}_Hidden{h_dim}_Dropout{drop}.pdf"
        save_path = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path, format='pdf')
        plt.close()

def plot_master_combined(grouped_data, model_name, output_dir):
    """
    Generates ONE Master PDF with all configurations overlaid (Comparison View).
    """
    print(f"   -> Generating MASTER combined plot...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Generate distinct colors for each configuration
    unique_keys = sorted(grouped_data.keys()) # Sort to keep legend consistent
    num_keys = len(unique_keys)
    colors = cm.tab10(np.linspace(0, 1, num_keys)) if num_keys <= 10 else cm.rainbow(np.linspace(0, 1, num_keys))

    for i, key in enumerate(unique_keys):
        h_dim, drop = key
        points = grouped_data[key]
        
        lrs = [p[0] for p in points]
        corrs = [p[1] for p in points]
        
        label_text = f"H:{h_dim}, D:{drop}"
        
        # Plot line and points
        ax.plot(lrs, corrs, color=colors[i], linestyle='--', alpha=0.7, label=label_text)
        ax.scatter(lrs, corrs, color=colors[i], s=80, edgecolors='white', zorder=3)

    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate (Log Scale)', fontsize=12)
    ax.set_ylabel('Validation Spearman Correlation', fontsize=12)
    ax.set_title(f"{model_name}: All Configurations Combined", fontsize=15)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    
    # Place legend outside if too crowded, or best location inside
    ax.legend(title="Configuration", fontsize=10, loc='best', fancybox=True, framealpha=0.8)

    filename = f"{model_name}_ALL_COMBINED.pdf"
    save_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()
    print(f"   -> Saved Master Plot: {filename}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    cnn_config = [
        ('CNN_search_results.json', 0.0),
        ('CNN_search_results_t2 copy.json', 0.0),
        ('CNN_search_results_t3 copy.json', 0.0),
        ('CNN_search_results_t4 copy.json', 0.0),
        ('CNN_search_results_t5 copy.json', 0.0)
    ]
    
    lstm_config = [
        ('LSTM_search_results.json', 0.0),
        ('LSTM_search_results_t2 copy.json', 0.0),
        ('LSTM_search_results_t3 copy.json', 0.0),
        ('LSTM_search_results_t4 copy.json', 0.0),
        ('LSTM_search_results_t5 copy.json', 0.0)
    ]

    output_dir = 'Evaluation_Outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- PROCESS CNN ---
    print("\n--- Processing CNN Data ---")
    cnn_results = load_and_filter_data(cnn_config)
    if cnn_results:
        cnn_grouped = organize_data(cnn_results)
        plot_individual_configs(cnn_grouped, "CNN", output_dir)
        plot_master_combined(cnn_grouped, "CNN", output_dir)

    # --- PROCESS LSTM ---
    print("\n--- Processing LSTM Data ---")
    lstm_results = load_and_filter_data(lstm_config)
    if lstm_results:
        lstm_grouped = organize_data(lstm_results)
        plot_individual_configs(lstm_grouped, "LSTM", output_dir)
        plot_master_combined(lstm_grouped, "LSTM", output_dir)
    
    print(f"\nDone! All plots saved in: {output_dir}")