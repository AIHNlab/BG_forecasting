import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), 'CG-EGA'))
from cg_ega.cg_ega import CG_EGA
import pandas as pd

def plot_and_evaluate_horizon(pred_horizon, actual_horizon, std_horizon, participant, horizon, config,
                             hypo_threshold=70, hyper_threshold=180):
    """
    Plots predictions, actuals, confidence intervals, and computes RMSE for a given horizon.
    Also computes RMSE for hypo-, hyper-, and normoglycemic ranges.
    Returns a dict with RMSEs and saves the plot.
    """
    
    # Save plot to a file with confidence intervals
    evaluation_path = os.path.dirname(__file__)+os.sep+os.path.join(config['run_config']['experiment_path'], 'evaluation', str(participant), f'horizon_{horizon}')
    os.makedirs(evaluation_path, exist_ok=True)
    fig = plt.figure(figsize=(25, 5))

    # Plot actual values and predictions
    plt.plot(actual_horizon, label='Actuals', linestyle='-', linewidth=2, color='blue', alpha=0.7)
    plt.plot(pred_horizon, label='Predictions', linestyle='--', linewidth=1, color='red', alpha=0.7)

    # Add confidence intervals (±2 standard deviations for 95% confidence)
    plt.fill_between(
        range(len(pred_horizon)),
        pred_horizon - 2 * std_horizon,
        pred_horizon + 2 * std_horizon,
        color='red',
        alpha=0.2,
        label='95% Confidence Interval'
    )

    plt.legend()
    plt.title(f'Participant {participant} - Horizon {horizon}')
    plt.savefig(os.path.join(evaluation_path, 'plot.png'))
    plt.close(fig)

    # Calculate RMSE overall - only use valid (non-NaN) pairs
    valid_mask = ~np.isnan(actual_horizon) & ~np.isnan(pred_horizon)
    rmse = np.sqrt(np.mean((pred_horizon[valid_mask] - actual_horizon[valid_mask]) ** 2)) if np.any(valid_mask) else np.nan
    mae = np.mean(np.abs(pred_horizon[valid_mask] - actual_horizon[valid_mask])) if np.any(valid_mask) else np.nan

    # Calculate RMSE for different glycemic ranges
    hypo_mask = (actual_horizon < hypo_threshold) & (~np.isnan(actual_horizon)) & (~np.isnan(pred_horizon))
    hyper_mask = (actual_horizon > hyper_threshold) & (~np.isnan(actual_horizon)) & (~np.isnan(pred_horizon))
    normo_mask = ((actual_horizon >= hypo_threshold) & (actual_horizon <= hyper_threshold) & 
                 (~np.isnan(actual_horizon)) & (~np.isnan(pred_horizon)))

    # Print counts for debugging
    print(f"Participant {participant}, Horizon {horizon}: Valid points: {np.sum(valid_mask)}, " 
          f"Hypo: {np.sum(hypo_mask)}, Hyper: {np.sum(hyper_mask)}, Normo: {np.sum(normo_mask)}")

    rmse_hypo = np.sqrt(np.mean((pred_horizon[hypo_mask] - actual_horizon[hypo_mask]) ** 2)) if np.any(hypo_mask) else np.nan
    rmse_hyper = np.sqrt(np.mean((pred_horizon[hyper_mask] - actual_horizon[hyper_mask]) ** 2)) if np.any(hyper_mask) else np.nan
    rmse_normo = np.sqrt(np.mean((pred_horizon[normo_mask] - actual_horizon[normo_mask]) ** 2)) if np.any(normo_mask) else np.nan

    mae_hypo = np.mean(np.abs(pred_horizon[hypo_mask] - actual_horizon[hypo_mask])) if np.any(hypo_mask) else np.nan
    mae_hyper = np.mean(np.abs(pred_horizon[hyper_mask] - actual_horizon[hyper_mask])) if np.any(hyper_mask) else np.nan
    mae_normo = np.mean(np.abs(pred_horizon[normo_mask] - actual_horizon[normo_mask])) if np.any(normo_mask) else np.nan

    return {
        "rmse": rmse,
        "rmse_hypo": rmse_hypo,
        "rmse_hyper": rmse_hyper,
        "rmse_normo": rmse_normo,
        "mae": mae,
        "mae_hypo": mae_hypo,
        "mae_hyper": mae_hyper,
        "mae_normo": mae_normo
    }

def evaluate_cg_ega_horizon(pred_horizon, actual_horizon, participant, horizon, config, freq=5, plot_day=None):
    """
    Computes CG-EGA metrics for a single forecast horizon.
    Optionally plots and saves the CG-EGA plot for a given day.
    Returns (AP, BE, EP) rates.
    """
    # Prepare results DataFrame
    results = pd.DataFrame({
        "y_true": actual_horizon.flatten(),
        "y_pred": pred_horizon.flatten()
    })
    cg_ega = CG_EGA(results, freq)
    ap, be, ep = cg_ega.reduced()
    print(f"Participant {participant} - Horizon {horizon} CG-EGA: AP={ap:.3f}, BE={be:.3f}, EP={ep:.3f}")
    if plot_day is not None:
        evaluation_path = os.path.dirname(__file__)+os.sep+os.path.join(config['run_config']['experiment_path'], 'evaluation', str(participant), f'horizon_{horizon}')
        os.makedirs(evaluation_path, exist_ok=True)
        try:
            cg_ega.plot(day=plot_day)
        except Exception as e:
            print(f"Warning: CG-EGA plot failed for participant {participant}, horizon {horizon}, day {plot_day}: {e}")
        plt.savefig(os.path.join(evaluation_path, f'cg_ega_day{plot_day}.png'))
        plt.close()
    return ap, be, ep

def evaluate_uncertainty_calibration(y_true, mu, sigma, participant, horizon, config, confidence_levels=None, plot=True):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    if confidence_levels is None:
        confidence_levels = np.linspace(0.5, 0.99, 10)

    picp_list = []
    calibration_errors = []

    for alpha in confidence_levels:
        z = norm.ppf(1 - (1 - alpha) / 2)
        lower = mu - z * sigma
        upper = mu + z * sigma
        covered = (y_true >= lower) & (y_true <= upper)
        picp = covered.mean()
        picp_list.append(picp)
        calibration_errors.append(abs(picp - alpha))

    picp_array = np.array(picp_list)
    calibration_errors = np.array(calibration_errors)
    confidence_levels = np.array(confidence_levels)
    pice = np.mean(calibration_errors)

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(confidence_levels, picp_array, marker='o', label='Empirical PICP')
        ax[0].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal calibration')
        ax[0].set_xlabel('Predicted Confidence Level')
        ax[0].set_ylabel('Empirical Coverage (PICP)')
        ax[0].set_title('Calibration Curve')
        ax[0].legend()
        ax[0].grid(True)

        ax[1].bar([f"{int(c*100)}%" for c in confidence_levels], calibration_errors)
        ax[1].set_ylabel('Calibration Error')
        ax[1].set_xlabel('Confidence Level')
        ax[1].set_title('Per-Confidence-Level Calibration Error')
        ax[1].axhline(y=0, color='gray', linestyle='--')
        ax[1].grid(True)

        plt.tight_layout()
        evaluation_path = os.path.dirname(__file__)+os.sep+os.path.join(config['run_config']['experiment_path'], 'evaluation', str(participant), f'horizon_{horizon}')
        os.makedirs(os.path.dirname(evaluation_path), exist_ok=True)
        plt.savefig(evaluation_path + '/calibration_plot.png')
        plt.close()

    return {
        'confidence_levels': confidence_levels,
        'picp': picp_array,
        'calibration_error_per_level': calibration_errors,
        'PICE': pice
    }

def create_calibration_analysis(calibration_df, evaluation_path, config):
    """
    Create comprehensive calibration analysis including statistics and plots
    
    Args:
        calibration_df: DataFrame with calibration results for all participants and horizons
        evaluation_path: Path to save results
        config: Configuration dictionary
    """
    import json
    
    # Create a more structured JSON summary
    calibration_stats_json = {}
    
    # Statistics by horizon
    calibration_stats_json['by_horizon'] = {}
    for horizon in calibration_df['Horizon'].unique():
        horizon_data = calibration_df[calibration_df['Horizon'] == horizon].drop(columns=['Participant', 'Horizon'])
        
        calibration_stats_json['by_horizon'][str(horizon)] = {
            'pice': {
                'mean': float(horizon_data['PICE'].mean()),
                'std': float(horizon_data['PICE'].std()),
                'median': float(horizon_data['PICE'].median()),
                'min': float(horizon_data['PICE'].min()),
                'max': float(horizon_data['PICE'].max()),
                'count': int(horizon_data['PICE'].count())
            }
        }
        
        # Add PICP and calibration error stats for each confidence level
        picp_cols = [col for col in horizon_data.columns if col.startswith('PICP_')]
        calerr_cols = [col for col in horizon_data.columns if col.startswith('CALERR_')]
        
        if picp_cols:
            calibration_stats_json['by_horizon'][str(horizon)]['picp_by_level'] = {}
            for col in picp_cols:
                level = col.replace('PICP_', '')
                calibration_stats_json['by_horizon'][str(horizon)]['picp_by_level'][level] = {
                    'mean': float(horizon_data[col].mean()),
                    'std': float(horizon_data[col].std()),
                    'median': float(horizon_data[col].median())
                }
        
        if calerr_cols:
            calibration_stats_json['by_horizon'][str(horizon)]['calibration_error_by_level'] = {}
            for col in calerr_cols:
                level = col.replace('CALERR_', '')
                calibration_stats_json['by_horizon'][str(horizon)]['calibration_error_by_level'][level] = {
                    'mean': float(horizon_data[col].mean()),
                    'std': float(horizon_data[col].std()),
                    'median': float(horizon_data[col].median())
                }
    
    # Overall statistics (across all horizons)
    overall_data = calibration_df.drop(columns=['Participant', 'Horizon'])
    calibration_stats_json['overall'] = {
        'pice': {
            'mean': float(overall_data['PICE'].mean()),
            'std': float(overall_data['PICE'].std()),
            'median': float(overall_data['PICE'].median()),
            'min': float(overall_data['PICE'].min()),
            'max': float(overall_data['PICE'].max()),
            'count': int(overall_data['PICE'].count())
        }
    }
    
    # Add overall PICP and calibration error stats
    picp_cols = [col for col in overall_data.columns if col.startswith('PICP_')]
    calerr_cols = [col for col in overall_data.columns if col.startswith('CALERR_')]
    
    if picp_cols:
        calibration_stats_json['overall']['picp_by_level'] = {}
        for col in picp_cols:
            level = col.replace('PICP_', '')
            calibration_stats_json['overall']['picp_by_level'][level] = {
                'mean': float(overall_data[col].mean()),
                'std': float(overall_data[col].std()),
                'median': float(overall_data[col].median())
            }
    
    if calerr_cols:
        calibration_stats_json['overall']['calibration_error_by_level'] = {}
        for col in calerr_cols:
            level = col.replace('CALERR_', '')
            calibration_stats_json['overall']['calibration_error_by_level'][level] = {
                'mean': float(overall_data[col].mean()),
                'std': float(overall_data[col].std()),
                'median': float(overall_data[col].median())
            }
    
    # Save the structured JSON
    with open(os.path.join(evaluation_path, 'calibration_summary_stats.json'), 'w') as f:
        json.dump(calibration_stats_json, f, indent=4)
    
    # Also save the raw calibration summary as CSV (keep this for completeness)
    calibration_df.to_csv(os.path.join(evaluation_path, 'calibration_summary.csv'), index=False)

    # Create aggregated calibration plot
    plot_aggregated_calibration(calibration_df, evaluation_path, config)

    print("\n=== Calibration Summary Statistics ===")
    print(f"Overall PICE: {calibration_stats_json['overall']['pice']['mean']:.4f} ± {calibration_stats_json['overall']['pice']['std']:.4f}")
    
    # Print horizon-specific PICE
    for horizon, stats in calibration_stats_json['by_horizon'].items():
        print(f"Horizon {horizon} PICE: {stats['pice']['mean']:.4f} ± {stats['pice']['std']:.4f}")
    
    return calibration_stats_json

def plot_aggregated_calibration(calibration_df, evaluation_path, config):
    """
    Create aggregated calibration plots showing overall performance across all participants and horizons
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Get all PICP and CALERR columns
    picp_cols = [col for col in calibration_df.columns if col.startswith('PICP_')]
    calerr_cols = [col for col in calibration_df.columns if col.startswith('CALERR_')]
    
    if not picp_cols or not calerr_cols:
        print("Warning: No PICP or CALERR columns found for aggregated calibration plot")
        return
    
    # Extract confidence levels from column names
    confidence_levels = np.array([int(col.replace('PICP_', '')) / 100.0 for col in picp_cols])
    sort_idx = np.argsort(confidence_levels)
    confidence_levels = confidence_levels[sort_idx]
    picp_cols = [picp_cols[i] for i in sort_idx]
    calerr_cols = [calerr_cols[i] for i in sort_idx]
    
    # Calculate aggregated statistics
    mean_picp = calibration_df[picp_cols].mean().values
    std_picp = calibration_df[picp_cols].std().values
    mean_calerr = calibration_df[calerr_cols].mean().values
    std_calerr = calibration_df[calerr_cols].std().values
    
    # Calculate overall PICE
    overall_pice = calibration_df['PICE'].mean()
    overall_pice_std = calibration_df['PICE'].std()
    
    # Create the aggregated plot (same as before)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Aggregated Calibration Curve with error bars
    ax1 = axes[0, 0]
    ax1.errorbar(confidence_levels, mean_picp, yerr=std_picp, marker='o', capsize=5, 
                linewidth=2, markersize=8, label='Empirical PICP (Mean ± Std)')
    ax1.plot([0, 1], [0, 1], linestyle='--', color='red', linewidth=2, 
            label='Perfect Calibration', alpha=0.7)
    ax1.fill_between(confidence_levels, confidence_levels, mean_picp, 
                    alpha=0.3, color='lightblue', label='Calibration Gap')
    ax1.set_xlabel('Predicted Confidence Level')
    ax1.set_ylabel('Empirical Coverage (PICP)')
    ax1.set_title('Aggregated Calibration Curve\n(All Participants & Horizons)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Add PICE annotation
    ax1.text(0.05, 0.95, f'Overall PICE: {overall_pice:.4f} ± {overall_pice_std:.4f}', 
            transform=ax1.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            fontsize=12, fontweight='bold')
    
    # 2. Calibration Error by Confidence Level
    ax2 = axes[0, 1]
    x_pos = np.arange(len(confidence_levels))
    bars = ax2.bar(x_pos, mean_calerr, yerr=std_calerr, capsize=5, 
                  alpha=0.8, color='coral')
    ax2.set_ylabel('Calibration Error |PICP - α|')
    ax2.set_xlabel('Confidence Level')
    ax2.set_title('Aggregated Calibration Error\nby Confidence Level')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{int(c*100)}%" for c in confidence_levels], rotation=45)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax2.axhline(y=overall_pice, color='red', linestyle='-', alpha=0.7, 
               label=f'Mean PICE: {overall_pice:.4f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. PICE distribution by horizon
    ax3 = axes[1, 0]
    horizons = sorted(calibration_df['Horizon'].unique())
    pice_by_horizon = [calibration_df[calibration_df['Horizon'] == h]['PICE'].values for h in horizons]
    
    box_plot = ax3.boxplot(pice_by_horizon, labels=[f"H{h}" for h in horizons], patch_artist=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(horizons)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('PICE')
    ax3.set_xlabel('Forecast Horizon')
    ax3.set_title('PICE Distribution by Horizon')
    ax3.grid(True, alpha=0.3)
    
    # 4. Calibration performance heatmap
    ax4 = axes[1, 1]
    
    # Create heatmap data: horizons vs confidence levels
    heatmap_data = np.zeros((len(horizons), len(confidence_levels)))
    for i, horizon in enumerate(horizons):
        horizon_data = calibration_df[calibration_df['Horizon'] == horizon]
        for j, col in enumerate(calerr_cols):
            heatmap_data[i, j] = horizon_data[col].mean()
    
    im = ax4.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto', interpolation='nearest')
    ax4.set_xticks(range(len(confidence_levels)))
    ax4.set_xticklabels([f"{int(c*100)}%" for c in confidence_levels], rotation=45)
    ax4.set_yticks(range(len(horizons)))
    ax4.set_yticklabels([f"H{h}" for h in horizons])
    ax4.set_xlabel('Confidence Level')
    ax4.set_ylabel('Forecast Horizon')
    ax4.set_title('Calibration Error Heatmap\n(Darker = Better Calibrated)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
    cbar.set_label('Mean Calibration Error')
    
    # Add text annotations on heatmap
    for i in range(len(horizons)):
        for j in range(len(confidence_levels)):
            text = ax4.text(j, i, f'{heatmap_data[i, j]:.3f}', 
                          ha="center", va="center", color="white" if heatmap_data[i, j] > np.max(heatmap_data)/2 else "black",
                          fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(evaluation_path, 'aggregated_calibration_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a simplified version with just the main calibration curve
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.errorbar(confidence_levels, mean_picp, yerr=std_picp, marker='o', capsize=5, 
               linewidth=3, markersize=10, label=f'Model PICP (PICE: {overall_pice:.4f})')
    ax.plot([0, 1], [0, 1], linestyle='--', color='red', linewidth=3, 
           label='Perfect Calibration', alpha=0.8)
    ax.fill_between(confidence_levels, confidence_levels, mean_picp, 
                   alpha=0.3, color='lightblue')
    ax.set_xlabel('Predicted Confidence Level', fontsize=14)
    ax.set_ylabel('Empirical Coverage (PICP)', fontsize=14)
    ax.set_title('Model Calibration Performance\n(Aggregated across all participants and horizons)', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(evaluation_path, 'calibration_curve_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # CREATE HORIZON-SPECIFIC PLOTS
    # Create horizon-specific directory
    horizon_plots_path = os.path.join(evaluation_path, 'calibration_by_horizon')
    os.makedirs(horizon_plots_path, exist_ok=True)
    
    # Plot for each horizon individually
    for horizon in horizons:
        horizon_data = calibration_df[calibration_df['Horizon'] == horizon]
        
        if len(horizon_data) == 0:
            continue
            
        # Calculate statistics for this horizon
        mean_picp_h = horizon_data[picp_cols].mean().values
        std_picp_h = horizon_data[picp_cols].std().values
        mean_calerr_h = horizon_data[calerr_cols].mean().values
        std_calerr_h = horizon_data[calerr_cols].std().values
        horizon_pice = horizon_data['PICE'].mean()
        horizon_pice_std = horizon_data['PICE'].std()
        
        # Create 2x2 subplot for this horizon
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Calibration Curve
        ax1 = axes[0, 0]
        ax1.errorbar(confidence_levels, mean_picp_h, yerr=std_picp_h, marker='o', capsize=5, 
                    linewidth=2, markersize=8, label=f'Horizon {horizon} PICP')
        ax1.plot([0, 1], [0, 1], linestyle='--', color='red', linewidth=2, 
                label='Perfect Calibration', alpha=0.7)
        ax1.fill_between(confidence_levels, confidence_levels, mean_picp_h, 
                        alpha=0.3, color='lightblue', label='Calibration Gap')
        ax1.set_xlabel('Predicted Confidence Level')
        ax1.set_ylabel('Empirical Coverage (PICP)')
        ax1.set_title(f'Calibration Curve - Horizon {horizon}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        # Add PICE annotation
        ax1.text(0.05, 0.95, f'PICE: {horizon_pice:.4f} ± {horizon_pice_std:.4f}', 
                transform=ax1.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                fontsize=12, fontweight='bold')
        
        # 2. Calibration Error by Confidence Level
        ax2 = axes[0, 1]
        x_pos = np.arange(len(confidence_levels))
        bars = ax2.bar(x_pos, mean_calerr_h, yerr=std_calerr_h, capsize=5, 
                      alpha=0.8, color='coral')
        ax2.set_ylabel('Calibration Error |PICP - α|')
        ax2.set_xlabel('Confidence Level')
        ax2.set_title(f'Calibration Error - Horizon {horizon}')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"{int(c*100)}%" for c in confidence_levels], rotation=45)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax2.axhline(y=horizon_pice, color='red', linestyle='-', alpha=0.7, 
                   label=f'Mean PICE: {horizon_pice:.4f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. PICE distribution across participants
        ax3 = axes[1, 0]
        pice_values = horizon_data['PICE'].values
        if len(pice_values) > 1:
            ax3.hist(pice_values, bins=min(10, len(pice_values)), alpha=0.7, color='lightgreen', edgecolor='black')
        else:
            ax3.bar([0], [1], alpha=0.7, color='lightgreen', edgecolor='black')
            ax3.set_xticks([0])
            ax3.set_xticklabels([f'{pice_values[0]:.4f}'])
        ax3.axvline(horizon_pice, color='red', linestyle='--', linewidth=2, label=f'Mean: {horizon_pice:.4f}')
        if len(pice_values) > 1:
            ax3.axvline(np.median(pice_values), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(pice_values):.4f}')
        ax3.set_xlabel('PICE')
        ax3.set_ylabel('Count (Participants)')
        ax3.set_title(f'PICE Distribution - Horizon {horizon}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Individual participant PICP scatter
        ax4 = axes[1, 1]
        
        # Show individual participant PICP values as scatter points
        for i, conf_level in enumerate(confidence_levels):
            picp_values = horizon_data[picp_cols[i]].values
            x_scatter = np.full_like(picp_values, conf_level) + np.random.normal(0, 0.01, len(picp_values))
            ax4.scatter(x_scatter, picp_values, alpha=0.6, s=30, color='lightblue')
        
        # Overlay the mean line
        ax4.plot(confidence_levels, mean_picp_h, 'o-', color='blue', linewidth=3, 
                markersize=8, label='Mean PICP')
        ax4.plot([0, 1], [0, 1], '--', color='red', linewidth=2, 
                label='Perfect Calibration')
        
        ax4.set_xlabel('Predicted Confidence Level')
        ax4.set_ylabel('Empirical Coverage (PICP)')
        ax4.set_title(f'Individual Participant Calibration - Horizon {horizon}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        
        plt.suptitle(f'Calibration Analysis - Forecast Horizon {horizon}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(horizon_plots_path, f'calibration_horizon_{horizon}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a simple calibration curve for this horizon
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.errorbar(confidence_levels, mean_picp_h, yerr=std_picp_h, marker='o', capsize=5, 
                   linewidth=3, markersize=10, label=f'Horizon {horizon} PICP (PICE: {horizon_pice:.4f})')
        ax.plot([0, 1], [0, 1], linestyle='--', color='red', linewidth=3, 
               label='Perfect Calibration', alpha=0.8)
        ax.fill_between(confidence_levels, confidence_levels, mean_picp_h, 
                       alpha=0.3, color='lightblue')
        ax.set_xlabel('Predicted Confidence Level', fontsize=14)
        ax.set_ylabel('Empirical Coverage (PICP)', fontsize=14)
        ax.set_title(f'Calibration Performance - Forecast Horizon {horizon}\n({len(horizon_data)} participants)', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(horizon_plots_path, f'calibration_curve_horizon_{horizon}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create a comparison plot showing all horizons together
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(horizons)))
    
    for i, horizon in enumerate(horizons):
        horizon_data = calibration_df[calibration_df['Horizon'] == horizon]
        if len(horizon_data) == 0:
            continue
        mean_picp_h = horizon_data[picp_cols].mean().values
        std_picp_h = horizon_data[picp_cols].std().values
        horizon_pice = horizon_data['PICE'].mean()
        
        ax.errorbar(confidence_levels, mean_picp_h, yerr=std_picp_h, 
                   marker='o', capsize=3, linewidth=2, markersize=6, 
                   color=colors[i], label=f'Horizon {horizon} (PICE: {horizon_pice:.4f})')
    
    ax.plot([0, 1], [0, 1], linestyle='--', color='black', linewidth=2, 
           label='Perfect Calibration', alpha=0.8)
    ax.set_xlabel('Predicted Confidence Level', fontsize=14)
    ax.set_ylabel('Empirical Coverage (PICP)', fontsize=14)
    ax.set_title('Calibration Comparison Across Forecast Horizons', fontsize=16)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(evaluation_path, 'calibration_comparison_by_horizon.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Aggregated calibration plots saved to {evaluation_path}")
    print(f"Horizon-specific calibration plots saved to {horizon_plots_path}")