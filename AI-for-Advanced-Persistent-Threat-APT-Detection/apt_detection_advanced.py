

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA

# ==========================================
# 1. ADVANCED DATA GENERATION
# ==========================================
# Simulating a complex network environment with 25 features
# We introduce 'class_sep=0.8' to make the classes harder to separate (Realistic)
print("Generating High-Fidelity Synthetic Network Data...")
X, y = make_classification(n_samples=5000, n_features=25, n_informative=15,
                           n_redundant=5, n_clusters_per_class=2,
                           weights=[0.90, 0.10], class_sep=0.8, random_state=42)

# Feature Naming
feature_names = [f"Net_Flow_Feat_{i}" for i in range(25)]
feature_names[0] = "Flow_Duration_ms"
feature_names[1] = "Packet_Len_Var"
feature_names[2] = "IAT_Mean"

df = pd.DataFrame(X, columns=feature_names)
df['Label'] = y

# ==========================================
# 2. MODEL TRAINING
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Using 200 trees and limiting depth to prevent overfitting
rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
rf_model.fit(X_train, y_train)

# ==========================================
# 3. PREDICTION & METRICS
# ==========================================
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

print("\n--- Model Performance Report ---")
print(classification_report(y_test, y_pred))

# ==========================================
# 4. INNOVATIVE VISUALIZATIONS
# ==========================================
plt.figure(figsize=(20, 6))

# [A] Confusion Matrix
plt.subplot(1, 3, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='RdBu_r', cbar=False)
plt.title('Confusion Matrix\n(Precision vs Recall)', fontsize=14)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# [B] ROC Curve
plt.subplot(1, 3, 2)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='#d62728', lw=3, label=f'AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.fill_between(fpr, tpr, alpha=0.1, color='#d62728')
plt.title('ROC Curve (Sensitivity)', fontsize=14)
plt.legend(loc="lower right")

# [C] 3D Traffic Cluster Visualization (PCA)
# Projects 25 features down to 3D to visualize how the AI "sees" the attack
ax = plt.subplot(1, 3, 3, projection='3d')
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_test)
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y_test, cmap='coolwarm', alpha=0.6)
ax.set_title('3D AI Decision Space\n(Blue=Benign, Red=APT)', fontsize=14)
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')

plt.tight_layout()
plt.savefig('apt_analysis_results.png', dpi=300)
print("Plots saved to 'apt_analysis_results.png'")
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA

# 1. ADVANCED DATA GENERATION & SAVING

print("Generating High-Fidelity Synthetic Network Data...")
# We use 'class_sep=0.8' to make classes hard to separate (Realistic)
X, y = make_classification(n_samples=5000, n_features=25, n_informative=15,
                           n_redundant=5, n_clusters_per_class=2,
                           weights=[0.90, 0.10], class_sep=0.8, random_state=42)

# Feature Naming
feature_names = [f"Net_Flow_Feat_{i}" for i in range(25)]
feature_names[0] = "Flow_Duration_ms"
feature_names[1] = "Packet_Len_Var"
feature_names[2] = "IAT_Mean"

df = pd.DataFrame(X, columns=feature_names)
df['Label'] = y

# Saving Data
df.to_csv("CIC_IDS2017_Synthetic.csv", index=False)
print("✅ Data saved to 'CIC_IDS2017_Synthetic.csv'")

# 2. MODEL TRAINING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Using 200 trees and limiting depth to prevent overfitting
rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
rf_model.fit(X_train, y_train)

# 3. PREDICTION & METRICS
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

print("\n--- Model Performance Report ---")
print(classification_report(y_test, y_pred))

# 4. VISUALIZATIONS
plt.figure(figsize=(20, 6))

# [A] Confusion Matrix
plt.subplot(1, 3, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='RdBu_r', cbar=False)
plt.title('Confusion Matrix\n(Precision vs Recall)', fontsize=14)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# [B] ROC Curve
plt.subplot(1, 3, 2)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='#d62728', lw=3, label=f'AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.fill_between(fpr, tpr, alpha=0.1, color='#d62728')
plt.title('ROC Curve (Sensitivity)', fontsize=14)
plt.legend(loc="lower right")

# [C] 3D Traffic Cluster Visualization (PCA)
ax = plt.subplot(1, 3, 3, projection='3d')
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_test)
# Plotting a subset of points for clarity
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y_test, cmap='coolwarm', alpha=0.6)
ax.set_title('3D AI Decision Space\n(Blue=Benign, Red=APT)', fontsize=14)
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')

plt.tight_layout()
plt.savefig('apt_analysis_results.png', dpi=300)
print("Plots saved to 'apt_analysis_results.png'")
plt.show()
