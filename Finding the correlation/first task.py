import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Data extracted from the scatter plot
x = np.array([-10, -6, -5, -3, -1, 1, 3, 6, 7, 9])
y = np.array([5.50, 5, 3, 4, 1, 0, -2, -3, -4, -5])

# Calculate Pearson's correlation coefficient
correlation_coefficient, p_value = pearsonr(x, y)

# Calculate linear regression line for visualization
slope, intercept = np.polyfit(x, y, 1)
regression_line = slope * x + intercept

print("=" * 50)
print("PEARSON CORRELATION ANALYSIS RESULTS")
print("=" * 50)
print(f"Pearson's correlation coefficient (r): {correlation_coefficient:.3f}")
print(f"P-value: {p_value:.6f}")
print(f"Interpretation: {'Strong negative correlation' if correlation_coefficient < -0.7 else ''}")
print(f"Number of data points: {len(x)}")
print(f"Mean of x: {np.mean(x):.2f}")
print(f"Mean of y: {np.mean(y):.2f}")
print(f"Standard deviation of x: {np.std(x):.2f}")
print(f"Standard deviation of y: {np.std(y):.2f}")
print("=" * 50)

# Create visualization
plt.figure(figsize=(12, 8))

# Scatter plot of data points
plt.scatter(x, y, color='blue', s=150, edgecolors='black',
            linewidth=2, alpha=0.8, label='Data points', zorder=3)

# Add regression line
plt.plot(x, regression_line, color='red', linewidth=2.5,
         linestyle='--', alpha=0.7,
         label=f'Regression line (r = {correlation_coefficient:.3f})', zorder=2)

# Add grid and reference lines
plt.grid(True, alpha=0.3, linestyle='--', zorder=1)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)

# Annotate each data point with coordinates
for i, (xi, yi) in enumerate(zip(x, y)):
    plt.annotate(f'({xi}, {yi})',
                 (xi, yi),
                 xytext=(10, 10),
                 textcoords='offset points',
                 fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white",
                          edgecolor="gray",
                          alpha=0.8))

# Customize plot appearance
plt.xlabel('X Values', fontsize=14, fontweight='bold')
plt.ylabel('Y Values', fontsize=14, fontweight='bold')
plt.title('Scatter Plot with Pearson Correlation Analysis\n'
          f'r = {correlation_coefficient:.3f} (Strong Negative Correlation)',
          fontsize=16, fontweight='bold', pad=20)

# Add correlation information box
correlation_text = f"Pearson's r = {correlation_coefficient:.3f}\n"
correlation_text += f"p-value = {p_value:.6f}\n"
correlation_text += f"n = {len(x)} data points\n"
correlation_text += f"Equation: y = {slope:.3f}x + {intercept:.3f}"

plt.text(0.02, 0.98, correlation_text,
         transform=plt.gca().transAxes,
         fontsize=12,
         verticalalignment='top',
         bbox=dict(boxstyle="round",
                  facecolor="lightyellow",
                  edgecolor="orange",
                  alpha=0.9))

plt.legend(loc='upper left', fontsize=12)
plt.tight_layout()

# Save the visualization
plt.savefig('pearson_correlation_analysis.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'pearson_correlation_analysis.png'")
print("=" * 50)

# Display the plot
plt.show()