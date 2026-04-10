"""
Visual guide for interpreting calibration curves.
This creates example plots showing different calibration scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt

def create_example_calibration_plot(scenario_name, predicted, observed, save_name):
    """Create an example calibration plot for demonstration."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.7)
    
    # Actual calibration
    if scenario_name == "Well-Calibrated":
        color = 'green'
        marker = 'o'
    elif scenario_name == "Overconfident":
        color = 'red'
        marker = 's'
    else:  # Underconfident
        color = 'orange'
        marker = '^'
    
    ax.plot(predicted, observed, color=color, marker=marker, markersize=12, 
            linewidth=2, label='Model Calibration', alpha=0.8)
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=14, fontweight='bold')
    ax.set_ylabel('Observed Event Frequency', fontsize=14, fontweight='bold')
    ax.set_title(f'{scenario_name} Model', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Add interpretation text
    if scenario_name == "Well-Calibrated":
        text = "✓ Points close to diagonal\n✓ Predicted probabilities match reality\n✓ Trustworthy confidence estimates"
        bbox_color = 'lightgreen'
    elif scenario_name == "Overconfident":
        text = "✗ Points below diagonal\n✗ Model too confident\n✗ Predicted probabilities too high"
        bbox_color = 'lightcoral'
    else:
        text = "✗ Points above diagonal\n✗ Model too cautious\n✗ Predicted probabilities too low"
        bbox_color = 'lightsalmon'
    
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor=bbox_color, alpha=0.7, edgecolor='black', linewidth=2))
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created {save_name}")


# Create three example scenarios
print("Creating Calibration Curve Examples...")
print("=" * 60)

# Scenario 1: Well-Calibrated Model
predicted = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
observed = np.array([0.04, 0.14, 0.26, 0.34, 0.47, 0.54, 0.67, 0.74, 0.86, 0.94])
create_example_calibration_plot("Well-Calibrated", predicted, observed, 
                                "example_calibration_well_calibrated.png")

# Scenario 2: Overconfident Model
predicted = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
observed = np.array([0.03, 0.10, 0.18, 0.25, 0.32, 0.40, 0.48, 0.55, 0.62, 0.70])
create_example_calibration_plot("Overconfident", predicted, observed, 
                                "example_calibration_overconfident.png")

# Scenario 3: Underconfident Model
predicted = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
observed = np.array([0.08, 0.22, 0.35, 0.48, 0.60, 0.70, 0.78, 0.85, 0.92, 0.97])
create_example_calibration_plot("Underconfident", predicted, observed, 
                                "example_calibration_underconfident.png")

print("=" * 60)
print("\nExample calibration curves created!")
print("\nFiles created:")
print("  - example_calibration_well_calibrated.png")
print("  - example_calibration_overconfident.png")
print("  - example_calibration_underconfident.png")
print("\nThese show what different calibration scenarios look like.")
print("\nIn your actual experiments:")
print("  - Per-participant curves: experiments/<name>/evaluation/<participant>/alarms_threshold_X/")
print("  - Aggregate curves: experiments/<name>/evaluation/aggregate_calibration/")
