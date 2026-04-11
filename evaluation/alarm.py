"""Event-level alarm evaluation and calibration for hypo/hyperglycemia.

Provides functions to:

1. Threshold predicted probabilities into binary alarm signals.
2. Filter short spurious events and merge nearby detections.
3. Match predicted events to ground-truth events within a time window.
4. Compute precision, recall, detection lead-time, and daily false-alarm rate.
5. Build and plot calibration curves (reliability diagrams).
6. Aggregate calibration data across participants for population-level analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
import numpy as np
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import json
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

def compute_event_level_metrics(true_binary, pred_binary, sampling_interval=5, match_window_min=60):
    """Match predicted alarm events to true events and compute metrics.

    Events are identified by rising edges (0→1) in the binary signals.
    A predicted event is matched to the nearest true event that starts
    within ``match_window_min`` minutes ahead of the prediction.

    Args:
        true_binary: 1-D array of ground-truth binary event signal.
        pred_binary: 1-D array of predicted binary alarm signal.
        sampling_interval: Minutes between consecutive time-steps.
        match_window_min: Maximum look-ahead window (minutes) for matching.

    Returns:
        dict: ``precision``, ``recall``, ``average_detection_time_min``,
              ``daily_false_alarms``, and raw counts.
    """

    # Convert time to steps
    match_window = match_window_min // sampling_interval

    # Step 1: Find event start indices
    true_starts = np.where((true_binary[:-1] == 0) & (true_binary[1:] == 1))[0] + 1
    pred_starts = np.where((pred_binary[:-1] == 0) & (pred_binary[1:] == 1))[0] + 1

    matched_true_events = set()
    correct_warnings = []
    detection_times = []
    unmatched_warnings = []

    for pred_time in pred_starts:
        # Try to match to any true event starting within next 60 mins
        match_found = False
        for t_start in true_starts:
            if t_start >= pred_time and (t_start - pred_time) <= match_window:
                if t_start not in matched_true_events:  # one-to-one matching
                    matched_true_events.add(t_start)
                    correct_warnings.append(pred_time)
                    detection_times.append((t_start - pred_time) * sampling_interval)
                    match_found = True
                    break
        if not match_found:
            unmatched_warnings.append(pred_time)

    # Metric 1: Percentage of correct warnings
    precision = len(correct_warnings) / len(pred_starts) * 100 if len(pred_starts) > 0 else 0
    recall = len(correct_warnings) / len(true_starts) * 100 if len(true_starts) > 0 else 0

    # Metric 2: Event detection times (in minutes)
    avg_detection_time = np.mean(detection_times) if detection_times else None

    # Metric 3: Daily false alarms
    total_minutes = len(true_binary) * sampling_interval
    total_days = total_minutes / (60 * 24)
    false_alarms_per_day = len(unmatched_warnings) / total_days if total_days > 0 else None

    return {
        'precision(percentage of model warnings that were correct)': precision,
        'recall (percentage of true events that were warned)': recall,
        'average_detection_time_min': avg_detection_time,
        'daily_false_alarms': false_alarms_per_day,
        'n_correct': len(correct_warnings),
        'n_false': len(unmatched_warnings),
        'n_predicted': len(pred_starts),
        'n_true_events': len(true_starts)
    }

def filter_binary_events(binary, min_duration=3, min_separation=6):
    """Remove short-lived events and merge events separated by small gaps.

    Args:
        binary: 1-D binary array (0/1).
        min_duration: Minimum consecutive 1s for an event to be kept.
        min_separation: Gaps shorter than this are filled (events merged).

    Returns:
        np.ndarray: Smoothed binary event signal.
    """

    labeled, num_features = label(binary)
    filtered = np.zeros_like(binary)

    for region in range(1, num_features + 1):
        indices = np.where(labeled == region)[0]
        if len(indices) >= min_duration:
            filtered[indices] = 1

    # Fill short dips
    smoothed = filtered.copy()
    i = 0
    while i < len(smoothed):
        if smoothed[i] == 1:
            end = i
            while end < len(smoothed) and smoothed[end] == 1:
                end += 1
            dip_start, dip_end = end, end
            while dip_end < len(smoothed) and smoothed[dip_end] == 0:
                dip_end += 1
            if (dip_end - dip_start) < min_separation and dip_end < len(smoothed):
                smoothed[dip_start:dip_end] = 1
            i = dip_end
        else:
            i += 1
    return smoothed


def plot_glucose_and_probs(ax, glucose, hyper_probs, hypo_probs):
    ax.plot(glucose, label='Current Blood Glucose Levels', linestyle='-', linewidth=2, color='blue', alpha=0.7)
    ax.set_ylabel('Glucose (mg/dL)')
    ax.axhspan(0, 70, color='purple', alpha=0.1, label='Hypoglycemia Zone (<70 mg/dL)')
    ax.axhspan(70, 180, color='green', alpha=0.05, label='Normal Zone (70–180 mg/dL)')
    ax.set_ylim(0, max(400, np.nanmax(glucose)))
    ax.axhspan(180, ax.get_ylim()[1], color='orange', alpha=0.1, label='Hyperglycemia Zone (>180 mg/dL)')

    axb = ax.twinx()
    axb.plot(hyper_probs, label='Probability of Hyper', linestyle='--', linewidth=1, color='orange', alpha=0.7)
    axb.plot(hypo_probs, label='Probability of Hypo', linestyle='--', linewidth=1, color='red', alpha=0.7)
    axb.set_ylabel('Probability')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = axb.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')


def plot_filtered_events(ax, true_binary, pred_binary, threshold):
    ax.plot(true_binary, label='Sustained Hyperglycemia (Filtered)', color='orange', drawstyle='steps-post')
    ax.plot(pred_binary, label=f'Predicted Hyper Events (Filtered, p > {threshold})', 
            color='red', linestyle='--', drawstyle='steps-post')
    ax.set_ylabel('Hyperglycemia (1=Yes, 0=No)')
    ax.set_xlabel('Time Step')
    ax.set_yticks([0, 1])
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc='upper right')


def run_evaluation(actuals, hyper_probs, hypo_probs, participant, config, threshold=0.4, historic_context=None, required_samples=None):
    """Run full alarm evaluation for one participant at a given threshold.

    Applies a validity mask (based on ``historic_context`` channel
    requirements or a simple NaN sliding window), thresholds the
    predicted probabilities, filters events, computes event-level
    metrics, builds calibration curves, and saves plots.

    Args:
        actuals: Array of shape ``(N, H, C)`` with true BG values.
        hyper_probs: 1-D array of predicted hyperglycemia probabilities.
        hypo_probs: 1-D array of predicted hypoglycemia probabilities.
        participant: Participant identifier (used in file paths / titles).
        config: Full experiment config dict.
        threshold: Probability threshold for binary alarm decision.
        historic_context: Optional array of model input history for
            validity masking based on per-channel sample requirements.
        required_samples: Per-channel minimum valid sample counts.

    Returns:
        dict or None: Event-level metrics for hyper and hypo, or None
        if probability arrays are missing.
    """
    if hyper_probs is None or hypo_probs is None:
        return

    evaluation_path = _PROJECT_ROOT + os.sep + os.path.join(
        config['run_config']['experiment_path'], 'evaluation', str(participant), f'alarms_threshold_{threshold}'
    )
    os.makedirs(evaluation_path, exist_ok=True)

    current_bg_levels = actuals[:, 0].flatten()[:-1]
    if historic_context is not None:
        # Use the input given to the model for masking
        nan_mask = np.isnan(current_bg_levels)
        window = config['run_config'].get('required_samples_window', 24)

        # Get required samples for each input channel from config
        required_samples = config['run_config'].get('required_samples_during_test', [24, 1, 1])

        nan_window_mask = np.zeros_like(nan_mask, dtype=bool)
        nan_window_mask[:window] = True

        for i in range(window, len(nan_mask)):
            # Check input channel requirements using the model's input data
            should_mask = False

            if i < len(historic_context):
                #window_start = max(0, i - window)
                window_inputs = historic_context[i, -window:, :]  # Shape: [window_size, context, channels]

                # Check requirements for each input channel
                for channel_idx, required_count in enumerate(required_samples):
                    if channel_idx < window_inputs.shape[1]:  # Check if channel exists
                        # Count valid samples for this channel across the window and context
                        channel_data = window_inputs[:, channel_idx]  # Shape: [window_size, context]
                        valid_count = np.sum(~np.isnan(channel_data))

                        if valid_count < required_count:
                            should_mask = True
                            break

            nan_window_mask[i] = should_mask

        final_mask = nan_mask | nan_window_mask
    else:
        # Filter the arrays using the mask
        nan_mask = np.isnan(current_bg_levels)
        window = 24
        nan_window_mask = np.zeros_like(nan_mask, dtype=bool)
        for i in range(window, len(nan_mask)):
            if np.any(nan_mask[i-window:i]):
                nan_window_mask[i] = True
        final_mask = nan_mask | nan_window_mask

    hyper_probs_arr = np.array(hyper_probs[1:])
    hypo_probs_arr = np.array(hypo_probs[1:])
    hyper_probs_arr[final_mask] = np.nan
    hypo_probs_arr[final_mask] = np.nan

    current_bg_levels[final_mask] = np.nan

    fig, axs = plt.subplots(3, 1, figsize=(25, 15), sharex=True)

    # --- Hyperglycemia ---
    plot_glucose_and_probs(axs[0], current_bg_levels, hyper_probs_arr, hypo_probs_arr)
    axs[0].set_title(f'Participant {participant} - Alarm (Hyper/Hypo) - Threshold: {threshold}')

    bg_hyper_binary = (current_bg_levels > 180).astype(int)
    filtered_hyper = filter_binary_events(bg_hyper_binary)
    prob_hyper_binary = (hyper_probs_arr > threshold).astype(int)
    filtered_prob_hyper = filter_binary_events(prob_hyper_binary)
    plot_filtered_events(axs[1], filtered_hyper, filtered_prob_hyper, threshold)
    axs[1].set_title(f'Filtered Hyperglycemia Events + Thresholded Hyper Probabilities (Threshold: {threshold})')

    metrics_hyper = compute_event_level_metrics(
        true_binary=filtered_hyper,
        pred_binary=filtered_prob_hyper,
        sampling_interval=5,
        match_window_min=60
    )

    # --- Hypoglycemia ---
    bg_hypo_binary = (current_bg_levels < 70).astype(int)
    filtered_hypo = filter_binary_events(bg_hypo_binary)
    prob_hypo_binary = (hypo_probs_arr > threshold).astype(int)
    filtered_prob_hypo = filter_binary_events(prob_hypo_binary)
    plot_filtered_events(axs[2], filtered_hypo, filtered_prob_hypo, threshold)
    axs[2].set_title(f'Filtered Hypoglycemia Events + Thresholded Hypo Probabilities (Threshold: {threshold})')

    metrics_hypo = compute_event_level_metrics(
        true_binary=filtered_hypo,
        pred_binary=filtered_prob_hypo,
        sampling_interval=5,
        match_window_min=60
    )

    print(f"Participant {participant} event-level metrics (Hyperglycemia, Threshold {threshold}):")
    for key, val in metrics_hyper.items():
        print(f"{key}: {val:.2f}" if isinstance(val, float) else f"{key}: {val}")

    print(f"\nParticipant {participant} event-level metrics (Hypoglycemia, Threshold {threshold}):")
    for key, val in metrics_hypo.items():
        print(f"{key}: {val:.2f}" if isinstance(val, float) else f"{key}: {val}")

    plt.tight_layout()
    plt.savefig(os.path.join(evaluation_path, f'plot_hyper_hypo_threshold_{threshold}.png'))
    plt.close(fig)
    
    # Add threshold info to the metrics before saving
    metrics_hyper['threshold'] = threshold
    metrics_hypo['threshold'] = threshold
    
    with open(os.path.join(evaluation_path, f'event_level_metrics_hyper_threshold_{threshold}.json'), 'w') as f:
        json.dump(metrics_hyper, f, indent=2)
    with open(os.path.join(evaluation_path, f'event_level_metrics_hypo_threshold_{threshold}.json'), 'w') as f:
        json.dump(metrics_hypo, f, indent=2)

    return {
        'participant': participant,
        'threshold': threshold,
        'hyper': metrics_hyper,
        'hypo': metrics_hypo
    }


def compute_calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform'):
    """
    Compute calibration curve data.
    
    Args:
        y_true: Binary array of true event occurrences (0 or 1)
        y_prob: Array of predicted probabilities (0 to 1)
        n_bins: Number of bins for calibration curve
        strategy: 'uniform' or 'quantile' binning strategy
    
    Returns:
        Dictionary with calibration metrics and curve data
    """
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_prob))
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    
    if len(y_true) == 0:
        return None
    
    # Compute calibration curve using sklearn
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )
    
    # Compute calibration metrics
    brier_score = brier_score_loss(y_true, y_prob)
    
    # Expected Calibration Error (ECE)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_counts = []
    
    for i in range(n_bins):
        bin_mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:  # Last bin includes right edge
            bin_mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        
        bin_count = np.sum(bin_mask)
        bin_counts.append(int(bin_count))  # Convert to Python int
        
        if bin_count > 0:
            bin_accuracy = np.mean(y_true[bin_mask])
            bin_confidence = np.mean(y_prob[bin_mask])
            ece += (bin_count / len(y_true)) * np.abs(bin_accuracy - bin_confidence)
    
    # Maximum Calibration Error (MCE)
    mce = 0.0
    for i in range(len(fraction_of_positives)):
        mce = max(mce, np.abs(fraction_of_positives[i] - mean_predicted_value[i]))
    
    # Log loss (cross-entropy)
    try:
        logloss = log_loss(y_true, y_prob)
    except:
        logloss = None
    
    return {
        'fraction_of_positives': fraction_of_positives.tolist(),
        'mean_predicted_value': mean_predicted_value.tolist(),
        'bin_counts': bin_counts,
        'brier_score': float(brier_score),
        'expected_calibration_error': float(ece),
        'maximum_calibration_error': float(mce),
        'log_loss': float(logloss) if logloss is not None else None,
        'n_samples': int(len(y_true)),
        'n_positive': int(np.sum(y_true)),
        'positive_rate': float(np.mean(y_true))
    }


def plot_calibration_curve(calibration_data, event_type='hyperglycemia', save_path=None, show_distribution=True):
    """
    Plot calibration (reliability) curve.
    
    Args:
        calibration_data: Dictionary from compute_calibration_curve
        event_type: 'hyperglycemia' or 'hypoglycemia'
        save_path: Path to save the figure (optional)
        show_distribution: If True, shows sample distribution histogram (default: True)
    """
    if calibration_data is None:
        print(f"No calibration data available for {event_type}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot calibration curve
    ax.plot(
        calibration_data['mean_predicted_value'],
        calibration_data['fraction_of_positives'],
        marker='o',
        linewidth=2,
        markersize=8,
        label=f'{event_type.capitalize()} Calibration'
    )
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    
    # Conditionally add histogram of predictions as bars
    if show_distribution:
        ax2 = ax.twinx()
        bin_edges = np.linspace(0, 1, len(calibration_data['bin_counts']) + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax2.bar(
            bin_centers,
            calibration_data['bin_counts'],
            width=1.0/len(calibration_data['bin_counts']),
            alpha=0.3,
            color='gray',
            label='Sample Distribution'
        )
        ax2.set_ylabel('Count', fontsize=12)
    
    # Labels and title
    ax.set_xlabel('Mean Predicted Probability', fontsize=14)
    ax.set_ylabel('Fraction of Positives (Actual)', fontsize=14)
    ax.set_title(f'Calibration Curve - {event_type.capitalize()}', fontsize=16, fontweight='bold')
    
    # Add metrics text box
    metrics_text = (
        f"Brier Score: {calibration_data['brier_score']:.4f}\n"
        f"ECE: {calibration_data['expected_calibration_error']:.4f}\n"
        f"MCE: {calibration_data['maximum_calibration_error']:.4f}\n"
        f"Samples: {calibration_data['n_samples']}\n"
        f"Events: {calibration_data['n_positive']} ({calibration_data['positive_rate']*100:.2f}%)"
    )
    
    if calibration_data['log_loss'] is not None:
        metrics_text += f"\nLog Loss: {calibration_data['log_loss']:.4f}"
    
    #ax.text(
    #    0.05, 0.95,
    #    metrics_text,
    #    transform=ax.transAxes,
    #    fontsize=11,
    #    verticalalignment='top',
    #    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    #)
    
    # Grid and legend
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Combine legends
    if show_distribution:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=11)
    else:
        ax.legend(loc='lower right', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Calibration curve saved to: {save_path}")
    
    plt.close(fig)


def aggregate_calibration_data(all_participants_data, event_type='hyper'):
    """
    Aggregate calibration data across all participants.
    
    Args:
        all_participants_data: List of tuples (y_true, y_prob) for each participant
        event_type: 'hyper' or 'hypo'
    
    Returns:
        Aggregated y_true and y_prob arrays
    """
    all_y_true = []
    all_y_prob = []
    
    for y_true, y_prob in all_participants_data:
        if y_true is not None and y_prob is not None:
            # Remove NaN values
            mask = ~(np.isnan(y_true) | np.isnan(y_prob))
            all_y_true.append(y_true[mask])
            all_y_prob.append(y_prob[mask])
    
    if len(all_y_true) == 0:
        return None, None
    
    # Concatenate all participants
    all_y_true = np.concatenate(all_y_true)
    all_y_prob = np.concatenate(all_y_prob)
    
    return all_y_true, all_y_prob
