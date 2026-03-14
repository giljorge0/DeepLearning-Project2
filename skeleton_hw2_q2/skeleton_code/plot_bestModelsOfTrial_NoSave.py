import json
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_best_from_search_results(cnn_file, lstm_file, output_dir='Outputs_t3'):
    """
    Reads search result JSONs, finds the best model in each, and plots their history.
    """
    
    # Helper function to find the best experiment in a JSON list
    def load_best_history(filepath):
        print(f"Loading {filepath}...")
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            # Find the experiment dict with the highest 'best_val_corr'
            best_exp = max(results, key=lambda x: x['best_val_corr'])
            
            print(f"  Best Config Found: {best_exp['config']}")
            print(f"  Best Val Corr: {best_exp['best_val_corr']:.4f}")
            return best_exp['history']
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return None
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None

    # 1. Load Data
    cnn_hist = load_best_history(cnn_file)
    lstm_hist = load_best_history(lstm_file)

    if cnn_hist is None or lstm_hist is None:
        print("Could not load both files. Aborting plot.")
        return

    # 2. Prepare Epoch Ranges
    epochs_cnn = range(1, len(cnn_hist['train_losses']) + 1)
    epochs_lstm = range(1, len(lstm_hist['train_losses']) + 1)
    
    os.makedirs(output_dir, exist_ok=True)

    # --- Plot 1: Loss ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_cnn, cnn_hist['train_losses'], label='CNN Train Loss', color='blue', linestyle='--')
    plt.plot(epochs_cnn, cnn_hist['val_losses'], label='CNN Val Loss', color='blue')
    plt.plot(epochs_lstm, lstm_hist['train_losses'], label='LSTM Train Loss', color='red', linestyle='--')
    plt.plot(epochs_lstm, lstm_hist['val_losses'], label='LSTM Val Loss', color='red')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss (Best Models)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/best_models_loss_plot.png')
    plt.show()
    plt.close()

    # --- Plot 2: Accuracy (Spearman Correlation) ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_cnn, cnn_hist['train_correlations'], label='CNN Train Spearman', color='blue', linestyle='--')
    plt.plot(epochs_cnn, cnn_hist['val_correlations'], label='CNN Val Spearman', color='blue')
    plt.plot(epochs_lstm, lstm_hist['train_correlations'], label='LSTM Train Spearman', color='red', linestyle='--')
    plt.plot(epochs_lstm, lstm_hist['val_correlations'], label='LSTM Val Spearman', color='red')
    
    plt.xlabel('Epochs')
    plt.ylabel('Spearman Correlation')
    plt.title('Training and Validation Spearman Correlation (Best Models)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/best_models_accuracy_plot.png')
    plt.show()
    plt.close()
    
    print(f"Plots saved to {output_dir}")


cnn_path = 'Outputs/CNN_search_results copy.json'
lstm_path = 'Outputs/LSTM_search_results copy.json'

plot_best_from_search_results(cnn_path, lstm_path)