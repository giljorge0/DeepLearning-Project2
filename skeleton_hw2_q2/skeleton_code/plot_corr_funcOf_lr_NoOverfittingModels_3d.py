import json
import os
import glob
import numpy as np
import plotly.graph_objects as go
import math

def is_allowed(model_type, dropout, lr):
    """
    Checks if a specific Model/Dropout/LR combination is in the 'Non-Overfitting' whitelist.
    Uses math.isclose for robust float comparison.
    """
    # Define the whitelist based on your specific findings
    allowed_configs = {
        'CNN': {
            0.1: [], # Overfits everywhere
            0.2: [0.003, 0.005],
            0.3: [0.001, 0.0001, 0.003, 0.005, 0.00001, 0.00006]
        },
        'LSTM': {
            0.1: [0.0001, 0.00006],
            0.3: [0.00006],
            0.4: [0.0001, 0.00006]
        }
    }

    # 1. Check if model exists in rules
    if model_type not in allowed_configs:
        return False
    
    # 2. Check if dropout exists for that model
    # We loop to find a close match for dropout (floats)
    matched_drop = None
    for d in allowed_configs[model_type]:
        if math.isclose(dropout, d, abs_tol=1e-5):
            matched_drop = d
            break
            
    if matched_drop is None:
        return False

    # 3. Check if LR is in the allowed list for that dropout
    valid_lrs = allowed_configs[model_type][matched_drop]
    for valid_lr in valid_lrs:
        if math.isclose(lr, valid_lr, rel_tol=1e-5):
            return True
            
    return False

def plot_3d_interactive_filtered():
    # 1. Setup Output Directory
    output_dir = '3D_Interactive_Plots_Filtered'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 2. Define folders and pattern
    search_dirs = ['Outputs', 'Outputs_t2', 'Outputs_t3', 'Outputs_t4', 'Outputs_t5', "Outputs_t6"]
    file_pattern = '*copy.json'

    # 3. Data Storage
    plot_data = {'CNN': [], 'LSTM': []}

    print(f"Scanning for '{file_pattern}' in {search_dirs}...")

    # --- PHASE 1: LOAD AND FILTER DATA ---
    for folder in search_dirs:
        if folder and not os.path.exists(folder):
            continue
            
        search_path = os.path.join(folder, file_pattern) if folder else file_pattern
        found_files = glob.glob(search_path)
        
        for file_path in found_files:
            try:
                with open(file_path, 'r') as f:
                    content = json.load(f)
                
                if not isinstance(content, list):
                    continue

                is_lstm = "LSTM" in file_path
                model_type = "LSTM" if is_lstm else "CNN"

                for exp in content:
                    config = exp.get('config', {})
                    
                    # 1. H_DIM FILTER
                    h_dim = int(config.get('hidden_dim', -1))
                    if h_dim != 256:
                        continue

                    # 2. OVERFITTING FILTER (The Function above)
                    lr = float(config.get('learning_rate', -1))
                    drop = float(config.get('dropout', -1))
                    val_corr = float(exp.get('best_val_corr', 0.0))

                    if is_allowed(model_type, drop, lr):
                        plot_data[model_type].append((lr, drop, val_corr))

            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    # --- PHASE 2: GENERATE PLOTS ---
    
    for model_name, points in plot_data.items():
        if not points:
            print(f"No valid non-overfitting data found for {model_name} (H=256).")
            continue

        # Unpack data
        lrs = [p[0] for p in points]
        drops = [p[1] for p in points]
        corrs = [p[2] for p in points]

        # Create 3D Scatter Plot
        fig = go.Figure(data=[go.Scatter3d(
            x=lrs,
            y=drops,
            z=corrs,
            mode='markers',
            marker=dict(
                size=8,
                color=corrs,
                colorscale='Viridis',
                opacity=0.9,
                colorbar=dict(title='Correlation')
            ),
            # Hover text showing exactly what this point is
            text=[f"LR: {lr}<br>Drop: {d}<br>Corr: {c:.4f}" for lr, d, c in zip(lrs, drops, corrs)],
            hoverinfo='text'
        )])

        # Layout Settings
        fig.update_layout(
            title=f"Non-Overfitting Landscape: {model_name} (Hidden=256)",
            scene=dict(
                xaxis=dict(
                    type='log',
                    title='Learning Rate (Log Scale)',
                    tickformat='.1e'
                ),
                yaxis=dict(title='Dropout'),
                zaxis=dict(title='Validation Correlation')
            ),
            width=1000,
            height=800,
            margin=dict(r=20, b=10, l=10, t=40)
        )

        # Save as HTML
        filename = f"{model_name}_3D_Filtered.html"
        save_path = os.path.join(output_dir, filename)
        fig.write_html(save_path)
        
        print(f"Saved filtered interactive plot: {save_path}")
        print(f" -> Plotted {len(points)} points for {model_name}.")

    print(f"\nDone! Open the files in '{output_dir}'.")

if __name__ == "__main__":
    plot_3d_interactive_filtered()