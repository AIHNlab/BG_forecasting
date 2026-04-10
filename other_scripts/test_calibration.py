"""
Test script to demonstrate alarm calibration functionality.
This script shows how to use the new calibration functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from alarm_evaluation import compute_calibration_metrics, plot_calibration_curve

# Generate synthetic data for demonstration
np.random.seed(42)
n_samples = 5000

# Simulate predicted probabilities (somewhat calibrated but not perfect)
y_prob = np.random.beta(2, 5, n_samples)  # Beta distribution for probabilities

# Simulate actual outcomes based on probabilities with some noise
y_true = (y_prob + np.random.normal(0, 0.15, n_samples)) > 0.5
y_true = y_true.astype(float)

print("Testing Calibration Analysis")
print("=" * 60)
print(f"Total samples: {n_samples}")
print(f"Positive events: {y_true.sum()} ({y_true.mean()*100:.2f}%)")
print(f"Mean predicted probability: {y_prob.mean():.3f}")
print("=" * 60)

# Compute calibration metrics
calibration_metrics = compute_calibration_metrics(y_true, y_prob, n_bins=10)

if calibration_metrics is not None:
    print("\nCalibration Metrics:")
    print(f"  ECE (Expected Calibration Error): {calibration_metrics['ece']:.4f}")
    print(f"  MCE (Maximum Calibration Error): {calibration_metrics['mce']:.4f}")
    print(f"  Brier Score: {calibration_metrics['brier_score']:.4f}")
    print(f"  Valid samples: {calibration_metrics['n_samples']}")
    
    print("\nBin Statistics:")
    for i, bin_stat in enumerate(calibration_metrics['bin_stats']):
        print(f"  Bin {i+1}: [{bin_stat['bin_range'][0]:.1f}, {bin_stat['bin_range'][1]:.1f})")
        print(f"    Count: {bin_stat['count']}")
        print(f"    Mean Predicted: {bin_stat['mean_predicted']:.3f}")
        print(f"    Observed Frequency: {bin_stat['observed_frequency']:.3f}")
        print(f"    Calibration Error: {bin_stat['calibration_error']:.3f}")
    
    # Plot calibration curve
    plot_calibration_curve(
        calibration_metrics,
        'Test Calibration Curve - Synthetic Data',
        'test_calibration_curve.png',
        event_type='hyperglycemia'
    )
    print("\nCalibration curve saved as 'test_calibration_curve.png'")
    
    print("\n" + "=" * 60)
    print("Interpretation:")
    print("  - ECE close to 0 = well-calibrated")
    print("  - Points on diagonal (y=x) = perfect calibration")
    print("  - Points above diagonal = underconfident")
    print("  - Points below diagonal = overconfident")
    print("=" * 60)
else:
    print("Failed to compute calibration metrics")
