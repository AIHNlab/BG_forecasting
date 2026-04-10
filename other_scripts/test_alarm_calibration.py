"""
Test script for alarm calibration curves.
This demonstrates the calibration curve functionality with synthetic data.
"""

import numpy as np
import matplotlib.pyplot as plt
from alarm_evaluation import compute_calibration_curve, plot_calibration_curve, aggregate_calibration_data
import os
import json

def generate_synthetic_alarm_data(n_samples=10000, calibrated=True):
    """
    Generate synthetic alarm probability and event data.
    
    Args:
        n_samples: Number of samples to generate
        calibrated: If True, generates well-calibrated data; if False, poorly calibrated
    
    Returns:
        y_true, y_prob arrays
    """
    if calibrated:
        # Well-calibrated: actual event probability matches predicted probability
        y_prob = np.random.uniform(0, 1, n_samples)
        y_true = (np.random.uniform(0, 1, n_samples) < y_prob).astype(float)
    else:
        # Poorly calibrated: overconfident predictions
        base_prob = np.random.uniform(0, 0.5, n_samples)
        y_prob = np.minimum(base_prob * 2, 1.0)  # Overconfident
        y_true = (np.random.uniform(0, 1, n_samples) < base_prob).astype(float)
    
    return y_true, y_prob


def test_calibration_curves():
    """Test calibration curves with synthetic data."""
    
    # Create output directory
    output_dir = os.path.join('experiments', 'AlarmCalibration', 'test_calibration')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Testing Alarm Calibration Curves")
    print("=" * 60)
    
    # Test 1: Well-calibrated data
    print("\n1. Testing with well-calibrated data...")
    y_true_good, y_prob_good = generate_synthetic_alarm_data(n_samples=10000, calibrated=True)
    
    calibration_good = compute_calibration_curve(y_true_good, y_prob_good, n_bins=10)
    if calibration_good:
        print(f"   Brier Score: {calibration_good['brier_score']:.4f}")
        print(f"   Expected Calibration Error (ECE): {calibration_good['expected_calibration_error']:.4f}")
        print(f"   Maximum Calibration Error (MCE): {calibration_good['maximum_calibration_error']:.4f}")
        
        plot_path = os.path.join(output_dir, 'calibration_curve_well_calibrated.png')
        plot_calibration_curve(calibration_good, event_type='Well-Calibrated Test', save_path=plot_path)
        
        # Save metrics
        with open(os.path.join(output_dir, 'calibration_metrics_well_calibrated.json'), 'w') as f:
            json.dump(calibration_good, f, indent=4)
    
    # Test 2: Poorly calibrated data
    print("\n2. Testing with poorly calibrated (overconfident) data...")
    y_true_bad, y_prob_bad = generate_synthetic_alarm_data(n_samples=10000, calibrated=False)
    
    calibration_bad = compute_calibration_curve(y_true_bad, y_prob_bad, n_bins=10)
    if calibration_bad:
        print(f"   Brier Score: {calibration_bad['brier_score']:.4f}")
        print(f"   Expected Calibration Error (ECE): {calibration_bad['expected_calibration_error']:.4f}")
        print(f"   Maximum Calibration Error (MCE): {calibration_bad['maximum_calibration_error']:.4f}")
        
        plot_path = os.path.join(output_dir, 'calibration_curve_poorly_calibrated.png')
        plot_calibration_curve(calibration_bad, event_type='Poorly Calibrated Test', save_path=plot_path)
        
        # Save metrics
        with open(os.path.join(output_dir, 'calibration_metrics_poorly_calibrated.json'), 'w') as f:
            json.dump(calibration_bad, f, indent=4)
    
    # Test 3: Multi-participant aggregation
    print("\n3. Testing multi-participant aggregation...")
    participant_data = []
    for i in range(5):
        y_true, y_prob = generate_synthetic_alarm_data(n_samples=2000, calibrated=True)
        participant_data.append((y_true, y_prob))
    
    all_y_true, all_y_prob = aggregate_calibration_data(participant_data, event_type='hyper')
    if all_y_true is not None:
        print(f"   Aggregated {len(all_y_true)} samples from {len(participant_data)} participants")
        
        calibration_agg = compute_calibration_curve(all_y_true, all_y_prob, n_bins=10)
        if calibration_agg:
            print(f"   Brier Score: {calibration_agg['brier_score']:.4f}")
            print(f"   Expected Calibration Error (ECE): {calibration_agg['expected_calibration_error']:.4f}")
            
            plot_path = os.path.join(output_dir, 'calibration_curve_aggregated.png')
            plot_calibration_curve(calibration_agg, event_type='Aggregated Multi-Participant', save_path=plot_path)
    
    print("\n" + "=" * 60)
    print(f"Test complete! Results saved to: {output_dir}")
    print("=" * 60)
    
    # Print interpretation guide
    print("\n📊 Interpretation Guide:")
    print("  • Brier Score: Lower is better (0 = perfect, 1 = worst)")
    print("  • ECE (Expected Calibration Error): Lower is better (0 = perfect)")
    print("  • MCE (Maximum Calibration Error): Lower is better (0 = perfect)")
    print("  • Perfect calibration: Points lie on diagonal (y = x)")
    print("  • Above diagonal: Model is underconfident")
    print("  • Below diagonal: Model is overconfident")


if __name__ == "__main__":
    test_calibration_curves()
