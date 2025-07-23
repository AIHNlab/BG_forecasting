import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
import numpy as np
import os
import json

def compute_event_level_metrics(true_binary, pred_binary, sampling_interval=5, match_window_min=60):

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
    if hyper_probs is None or hypo_probs is None:
        return

    evaluation_path = os.path.dirname(__file__) + os.path.join(
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
        nan_mask = np.isnan(historic_context)
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