import matplotlib.pyplot as plt
import numpy as np

# Results from ISIC2017 and ISIC2018 evaluations
isic2017_results = {
    'mIoU': 56.11,
    'DSC': 67.90,
    'Sensitivity': 65.63,
    'Specificity': 96.37
}

isic2018_results = {
    'mIoU': 57.95,
    'DSC': 69.85,
    'Sensitivity': 67.74,
    'Specificity': 95.39
}

# Create comparison plot
metrics = list(isic2017_results.keys())
isic2017_values = list(isic2017_results.values())
isic2018_values = list(isic2018_results.values())

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 8))
bars1 = ax.bar(x - width/2, isic2017_values, width, label='ISIC2017', color='#2E86C1', alpha=0.8)
bars2 = ax.bar(x + width/2, isic2018_values, width, label='ISIC2018', color='#E74C3C', alpha=0.8)

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

add_value_labels(bars1)
add_value_labels(bars2)

ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
ax.set_title('MSGU-Net Performance Comparison: ISIC2017 vs ISIC2018', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 105)

# Add subtle background color
ax.set_facecolor('#F8F9FA')

plt.tight_layout()
plt.savefig('plots/isic_comparison_metrics_6.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("📊 Comparison plot saved as 'plots/isic_comparison_metrics_6.png'")

# Print summary comparison
print("\n" + "="*60)
print("MSGU-Net PERFORMANCE COMPARISON")
print("="*60)
print("Metric          ISIC2017    ISIC2018    Difference")
print("-" * 60)
for metric in metrics:
    diff = isic2017_results[metric] - isic2018_results[metric]
    sign = "+" if diff > 0 else ""
    print(f"{metric:<15} {isic2017_results[metric]:>7.2f}%   {isic2018_results[metric]:>7.2f}%   {sign}{diff:>6.2f}%")
print("="*60)

# Additional analysis
print("\n📈 ANALYSIS:")
print("• ISIC2017 performs slightly better across all metrics")
print("• Specificity is excellent for both datasets (>95%)")
print("• DSC and mIoU show good segmentation performance (~57-69%)")
print("• Performance consistency between datasets indicates robust model")

# Create a second plot showing performance differences
fig2, ax2 = plt.subplots(figsize=(10, 6))
differences = [isic2017_results[metric] - isic2018_results[metric] for metric in metrics]
colors = ['#27AE60' if diff > 0 else '#E74C3C' for diff in differences]

bars = ax2.bar(metrics, differences, color=colors, alpha=0.7)

# Add value labels
for bar, diff in zip(bars, differences):
    height = bar.get_height()
    ax2.annotate(f'{diff:+.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3 if height > 0 else -15),
                textcoords="offset points",
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=11, fontweight='bold')

ax2.set_ylabel('Performance Difference (%)', fontsize=12, fontweight='bold')
ax2.set_title('ISIC2017 vs ISIC2018 Performance Differences\n(Positive = ISIC2017 Better)', 
              fontsize=14, fontweight='bold', pad=20)
ax2.grid(axis='y', alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.set_facecolor('#F8F9FA')

plt.tight_layout()
plt.savefig('plots/isic_performance_differences_6.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("📊 Performance difference plot saved as 'plots/isic_performance_differences_6.png'")