import json
import os
import numpy as np
import matplotlib.pyplot as plt

def analyze_and_plot_champions():
    # 1. Define Directories and Model Types
    directories = ["Outputs"] + [f"Outputs_t{i}" for i in range(2, 7)]
    model_types = ["CNN", "LSTM"]

    # 2. Initialize tracking
    best_models = {
        "CNN": {"score": -1.0, "info": None, "history": None},
        "LSTM": {"score": -1.0, "info": None, "history": None}
    }

    # Helper function to safely get metrics from history
    def get_metric(h, keys):
        for k in keys:
            if k in h: return h[k]
        return []
    
    print(f"{'='*80}")
    print(f" SCANNING ALL DIRECTORIES FOR BEST CNN AND BEST LSTM")
    print(f"{'='*80}")

    # --- SEARCHING LOOP ---
    for folder in directories:
        if folder == "Outputs":
            suffix = "" 
        else:
            suffix = "_" + folder.split('_')[1] 
            
        for m_type in model_types:
            filename = f"{m_type}_search_results{suffix} copy.json"
            filepath = os.path.join(folder, filename)
            
            if not os.path.exists(filepath):
                continue

            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  [Error] Could not read {filepath}: {e}")
                continue
                
            print(f" -> Scanning {m_type} in: {folder}/{filename} ({len(data)} exps)")

            for i, exp in enumerate(data):
                history = exp.get('history', {})
                config = exp.get('config', {})
                
                v_corr = get_metric(history, ['val_correlations', 'val_correlation'])
                v_loss = get_metric(history, ['val_losses', 'val_loss'])
                
                if not v_corr:
                    continue
                
                # Find best epoch based on Max Correlation
                best_idx = np.argmax(v_corr)
                current_max_corr = v_corr[best_idx]
                
                # Check against Global Champion
                if current_max_corr > best_models[m_type]["score"]:
                    best_models[m_type]["score"] = current_max_corr
                    best_models[m_type]["history"] = history
                    best_models[m_type]["info"] = {
                        "Folder": folder,
                        "File": filename,
                        "Exp_Index": i,
                        "Best_Epoch": best_idx + 1,
                        "Best_Corr": current_max_corr,
                        # We only store scalar info here, full history is in ["history"]
                        "LR": config.get('learning_rate', 'N/A'),
                        "Dropout": config.get('dropout', 'N/A'),
                        "Hidden": config.get('hidden_dim', 'N/A')
                    }

    # --- REPORTING ---
    print(f"\n{'='*80}")
    print(f" 🏆 FINAL CHAMPIONS REPORT")
    print(f"{'='*80}")

    for m_type in model_types:
        best = best_models[m_type]
        info = best["info"]
        hist = best["history"]
        
        print(f"\n🔹 BEST {m_type} MODEL")
        if info is None:
            print("   No valid data found.")
        else:
            # Extract full lists to get specific values
            t_loss = get_metric(hist, ['train_losses', 'train_loss'])
            v_loss = get_metric(hist, ['val_losses', 'val_loss'])
            t_corr = get_metric(hist, ['train_correlations', 'train_correlation'])
            v_corr = get_metric(hist, ['val_correlations', 'val_correlation'])

            # Indices
            best_idx = info['Best_Epoch'] - 1
            final_idx = len(v_corr) - 1

            # --- PRINT GENERAL INFO ---
            print(f"   • Location:                {info['Folder']} / {info['File']} (Exp {info['Exp_Index']})")
            print(f"   • Hyperparameters:         LR: {info['LR']} | Drop: {info['Dropout']} | H: {info['Hidden']}")
            
            # --- PRINT BEST EPOCH METRICS ---
            print(f"   • [BEST EPOCH: {info['Best_Epoch']}]")
            print(f"       - Train Loss:          {t_loss[best_idx]:.5f}")
            print(f"       - Val Loss:            {v_loss[best_idx]:.5f}")
            print(f"       - Train Correlation:   {t_corr[best_idx]:.5f}")
            print(f"       - Val Correlation:     {v_corr[best_idx]:.5f}")

            # --- PRINT FINAL EPOCH METRICS ---
            print(f"   • [FINAL EPOCH: {len(v_corr)}]")
            print(f"       - Train Loss:          {t_loss[final_idx]:.5f}")
            print(f"       - Val Loss:            {v_loss[final_idx]:.5f}")
            print(f"       - Train Correlation:   {t_corr[final_idx]:.5f}")
            print(f"       - Val Correlation:     {v_corr[final_idx]:.5f}")

        print("-" * 60)

    # --- PLOTTING ---
    print("\nGenerating Separate Plots for Champions...")

    # Ensure output directory exists
    output_dir = "Evaluation_Outputs" 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f" -> Created directory: {output_dir}")

    for m_type in model_types:
        hist = best_models[m_type]["history"]
        info = best_models[m_type]["info"]

        if not hist:
            print(f"Skipping plots for {m_type} (No data found).")
            continue

        t_loss = get_metric(hist, ['train_losses', 'train_loss'])
        v_loss = get_metric(hist, ['val_losses', 'val_loss'])
        t_corr = get_metric(hist, ['train_correlations', 'train_correlation'])
        v_corr = get_metric(hist, ['val_correlations', 'val_correlation'])

        epochs = range(1, len(t_loss) + 1)
        line_color = 'blue' if m_type == "CNN" else 'red'
        
        # Get the specific epoch where the best result happened
        best_epoch_x = info['Best_Epoch']

        # --- PLOT 1: LOSS ---
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, t_loss, label=f'{m_type} Train Loss', color=line_color, linestyle='--', alpha=0.7)
        plt.plot(epochs, v_loss, label=f'{m_type} Val Loss', color=line_color, linewidth=2)
        
        # Vertical dashed line at best epoch
        plt.axvline(x=best_epoch_x, color='black', linestyle='--', alpha=0.6, label=f'Best Epoch ({best_epoch_x})')
        
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        # NO TITLE
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        filename_loss = f'{output_dir}/{m_type}_best_loss.pdf'
        plt.savefig(filename_loss)
        plt.close()
        print(f" -> Saved: {filename_loss}")

        # --- PLOT 2: ACCURACY (CORRELATION) ---
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, t_corr, label=f'{m_type} Train Spearman', color=line_color, linestyle='--', alpha=0.7)
        plt.plot(epochs, v_corr, label=f'{m_type} Val Spearman', color=line_color, linewidth=2)
        
        # Vertical dashed line at best epoch
        plt.axvline(x=best_epoch_x, color='black', linestyle='--', alpha=0.6, label=f'Best Epoch ({best_epoch_x})')
        
        plt.xlabel('Epochs')
        plt.ylabel('Spearman Correlation')
        # NO TITLE
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        filename_corr = f'{output_dir}/{m_type}_best_accuracy.pdf'
        plt.savefig(filename_corr)
        plt.close()
        print(f" -> Saved: {filename_corr}")

if __name__ == "__main__":
    analyze_and_plot_champions()