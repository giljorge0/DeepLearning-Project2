import json
import matplotlib.pyplot as plt
import os
import glob
from collections import defaultdict

def get_history_key(history, candidates):
    for key in candidates:
        if key in history:
            return history[key]
    return []

def plot_consolidated_v2():
    # 1. Setup Output Directories
    base_dir = 'Consolidated_Dropout_Plots'
    cnn_dir = os.path.join(base_dir, 'CNN_Plots_2')
    lstm_dir = os.path.join(base_dir, 'LSTM_Plots_2')

    for d in [cnn_dir, lstm_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    print(f"Created output directories in: {base_dir}")

    # 2. Define folders and pattern
    search_dirs = ['Outputs', 'Outputs_t2', 'Outputs_t3', 'Outputs_t4', 'Outputs_t5', 'Outputs_t6']
    file_pattern = '*copy.json'

    # 3. Data Structure
    # Nested Dictionary: data[Model_Type][(Learning_Rate, Hidden_Dim)][Dropout] = history
    experiments = defaultdict(lambda: defaultdict(dict))

    print(f"Scanning for '{file_pattern}' in {search_dirs}...")

    # --- PHASE 1: LOAD ALL DATA ---
    for folder in search_dirs:
        if folder and not os.path.exists(folder):
            continue
            
        search_path = os.path.join(folder, file_pattern) if folder else file_pattern
        found_files = glob.glob(search_path)
        
        for file_path in found_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if not isinstance(data, list):
                    continue

                is_lstm = "LSTM" in file_path
                model_type = "LSTM" if is_lstm else "CNN"

                for exp in data:
                    config = exp.get('config', {})
                    lr = float(config.get('learning_rate', -1))
                    h_dim = int(config.get('hidden_dim', -1))
                    drop = float(config.get('dropout', -1))
                    
                    # Key excludes dropout, because we want to group by LR and H
                    config_key = (lr, h_dim)
                    
                    if 'history' in exp and exp['history']:
                        experiments[model_type][config_key][drop] = exp['history']

            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    # --- PHASE 2: GENERATE PLOTS ---
    
    # --- CNN PLOTTING (Comparing Dr=0.2 vs Dr=0.3) ---
    cnn_count = 0
    for (lr, h_dim), dropout_runs in experiments['CNN'].items():
        if h_dim != 256:
            continue
            
        # Comparison Targets
        d1_val = 0.2
        d2_val = 0.3
        
        d1_hist = dropout_runs.get(d1_val)
        d2_hist = dropout_runs.get(d2_val)
        
        if not d1_hist and not d2_hist:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot Logic for Dropout 0.2 (BLUE)
        if d1_hist:
            t_loss = get_history_key(d1_hist, ['train_losses', 'train_loss', 'loss'])
            v_loss = get_history_key(d1_hist, ['val_losses', 'val_loss'])
            epochs = range(1, len(t_loss) + 1)
            ax1.plot(epochs, t_loss, label=f'Train (Dr={d1_val})', color='blue', linestyle='--', alpha=0.5)
            ax1.plot(epochs, v_loss, label=f'Val (Dr={d1_val})', color='blue', linewidth=2)
            
            t_corr = get_history_key(d1_hist, ['train_correlations', 'train_corrs', 'train_spearman'])
            v_corr = get_history_key(d1_hist, ['val_correlations', 'val_corrs', 'val_spearman'])
            if t_corr and v_corr:
                ax2.plot(epochs, t_corr, label=f'Train (Dr={d1_val})', color='blue', linestyle='--', alpha=0.5)
                ax2.plot(epochs, v_corr, label=f'Val (Dr={d1_val})', color='blue', linewidth=2)

        # Plot Logic for Dropout 0.3 (RED)
        if d2_hist:
            t_loss = get_history_key(d2_hist, ['train_losses', 'train_loss', 'loss'])
            v_loss = get_history_key(d2_hist, ['val_losses', 'val_loss'])
            epochs = range(1, len(t_loss) + 1)
            ax1.plot(epochs, t_loss, label=f'Train (Dr={d2_val})', color='red', linestyle='--', alpha=0.6)
            ax1.plot(epochs, v_loss, label=f'Val (Dr={d2_val})', color='red', linewidth=2)
            
            t_corr = get_history_key(d2_hist, ['train_correlations', 'train_corrs', 'train_spearman'])
            v_corr = get_history_key(d2_hist, ['val_correlations', 'val_corrs', 'val_spearman'])
            if t_corr and v_corr:
                ax2.plot(epochs, t_corr, label=f'Train (Dr={d2_val})', color='red', linestyle='--', alpha=0.6)
                ax2.plot(epochs, v_corr, label=f'Val (Dr={d2_val})', color='red', linewidth=2)

        # Formatting
        ax1.set_title("Loss (MSE)")
        ax1.set_xlabel("Epochs")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        ax2.set_title("Correlation (Spearman)")
        ax2.set_xlabel("Epochs")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.3)

        plt.suptitle(f"CNN Comparison | LR: {lr} | H: {h_dim}", fontsize=16)
        plt.tight_layout()
        
        # Save
        safe_lr = str(lr).replace('.', 'pt')
        filename = f"CNN_LR{safe_lr}_H{h_dim}.pdf"
        plt.savefig(os.path.join(cnn_dir, filename), format='pdf')
        plt.close()
        cnn_count += 1


    # --- LSTM PLOTTING (Comparing Dr=0.1 vs Dr=0.4) ---
    lstm_count = 0
    for (lr, h_dim), dropout_runs in experiments['LSTM'].items():
        if h_dim != 256:
            continue
            
        # Comparison Targets
        d1_val = 0.1
        d2_val = 0.4

        d1_hist = dropout_runs.get(d1_val)
        d2_hist = dropout_runs.get(d2_val)
        
        if not d1_hist and not d2_hist:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot Logic for Dropout 0.1 (BLUE)
        if d1_hist:
            t_loss = get_history_key(d1_hist, ['train_losses', 'train_loss', 'loss'])
            v_loss = get_history_key(d1_hist, ['val_losses', 'val_loss'])
            epochs = range(1, len(t_loss) + 1)
            ax1.plot(epochs, t_loss, label=f'Train (Dr={d1_val})', color='blue', linestyle='--', alpha=0.5)
            ax1.plot(epochs, v_loss, label=f'Val (Dr={d1_val})', color='blue', linewidth=2)
            
            t_corr = get_history_key(d1_hist, ['train_correlations', 'train_corrs', 'train_spearman'])
            v_corr = get_history_key(d1_hist, ['val_correlations', 'val_corrs', 'val_spearman'])
            if t_corr and v_corr:
                ax2.plot(epochs, t_corr, label=f'Train (Dr={d1_val})', color='blue', linestyle='--', alpha=0.5)
                ax2.plot(epochs, v_corr, label=f'Val (Dr={d1_val})', color='blue', linewidth=2)

        # Plot Logic for Dropout 0.4 (RED)
        if d2_hist:
            t_loss = get_history_key(d2_hist, ['train_losses', 'train_loss', 'loss'])
            v_loss = get_history_key(d2_hist, ['val_losses', 'val_loss'])
            epochs = range(1, len(t_loss) + 1)
            ax1.plot(epochs, t_loss, label=f'Train (Dr={d2_val})', color='red', linestyle='--', alpha=0.6)
            ax1.plot(epochs, v_loss, label=f'Val (Dr={d2_val})', color='red', linewidth=2)
            
            t_corr = get_history_key(d2_hist, ['train_correlations', 'train_corrs', 'train_spearman'])
            v_corr = get_history_key(d2_hist, ['val_correlations', 'val_corrs', 'val_spearman'])
            if t_corr and v_corr:
                ax2.plot(epochs, t_corr, label=f'Train (Dr={d2_val})', color='red', linestyle='--', alpha=0.6)
                ax2.plot(epochs, v_corr, label=f'Val (Dr={d2_val})', color='red', linewidth=2)

        # Formatting
        ax1.set_title("Loss (MSE)")
        ax1.set_xlabel("Epochs")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        ax2.set_title("Correlation (Spearman)")
        ax2.set_xlabel("Epochs")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.3)

        plt.suptitle(f"LSTM Comparison | LR: {lr} | H: {h_dim}", fontsize=16)
        plt.tight_layout()
        
        safe_lr = str(lr).replace('.', 'pt')
        filename = f"LSTM_LR{safe_lr}_H{h_dim}.pdf"
        plt.savefig(os.path.join(lstm_dir, filename), format='pdf')
        plt.close()
        lstm_count += 1

    print(f"DONE.\n -> Generated {cnn_count} CNN comparison plots in '{cnn_dir}'")
    print(f" -> Generated {lstm_count} LSTM comparison plots in '{lstm_dir}'")

if __name__ == "__main__":
    plot_consolidated_v2()