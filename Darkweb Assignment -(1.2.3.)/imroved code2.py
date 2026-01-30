import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report


# 1. DATA PREPARATION (Fixing previous mistakes)
def load_and_prepare_data(filepath):
    print("🔄 Loading and cleaning dataset...")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # Normalize casing to fix the "11 classes" issue in Label 2
    df['Label1'] = df['Label1'].astype(str).str.upper()
    df['Label2'] = df['Label2'].astype(str).str.upper()

    # Drop identifiers to prevent overfitting
    drop_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Src Port', 'Dst Port']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Handle Infinity values common in network flow data
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(df.median(numeric_only=True))
    return df


# 2. TRAINING & VISUALIZATION FUNCTION
def run_classification_with_viz(df, target_col, filename_prefix):
    print(f"\n🚀 Training: {target_col}")

    # Setup Features and Dynamic Labels
    X = df.drop(['Label1', 'Label2'], axis=1)
    y_raw = df[target_col]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = [str(cls) for cls in le.classes_]

    # Split & Scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Weights for imbalanced classes
    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw_dict = dict(enumerate(weights))

    # Architecture
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=512, validation_split=0.1,
              class_weight=cw_dict, verbose=1)

    # 3. GENERATE CONFUSION MATRIX
    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix: {target_col}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save visualization to file
    plt.savefig(f"confusion_matrix_{filename_prefix}.png")
    plt.close()

    print(f"\n--- {target_col} Report ---")
    print(classification_report(y_test, y_pred, target_names=class_names))


# EXECUTION
data = load_and_prepare_data("Darknet.csv")
run_classification_with_viz(data, "Label1", "label1")
run_classification_with_viz(data, "Label2", "label2")