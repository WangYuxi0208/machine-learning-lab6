import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset
X, y = load_digits(return_X_y=True)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train KNN (K=6)
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)

# Accuracy
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

# Classification report (English)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)

# Draw heatmap (beautiful blue theme, square cells)
plt.figure(figsize=(8, 8))
sns.heatmap(
    conf_mat,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=np.unique(y),
    yticklabels=np.unique(y),
    linewidths=0.3,
    linecolor='white',
    square=True
)

plt.title("KNN Confusion Matrix (K=6)", fontsize=14)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.tight_layout()

# Save figure to file
plt.savefig("confusion_matrix.png", dpi=300)

# Show image
plt.show()
