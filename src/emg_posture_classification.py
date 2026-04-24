"""Train and evaluate a CNN for EMG-based standing posture classification.

This script classifies four standing postures from segmented EMG windows and can
be run separately for SOL and FDB data.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras import Model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


POSTURE_NAMES = ["Bipedal", "One-leg", "Tandem", "Tiptoe"]


def set_random_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_mat_variable(file_path: Path, variable_name: str) -> np.ndarray:
    """Load a variable from a MATLAB .mat file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    mat = sio.loadmat(file_path)
    if variable_name not in mat:
        available = [key for key in mat.keys() if not key.startswith("__")]
        raise KeyError(
            f"Variable '{variable_name}' was not found in {file_path.name}. "
            f"Available variables: {available}"
        )
    return mat[variable_name]


def load_emg_dataset(
    data_dir: Path,
    muscle: str,
    n_channels: int = 64,
    n_samples: int = 512,
    window_label: str = "Time_W0.25O0.5",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train/test EMG data and labels for one muscle.

    Expected filenames follow the original project naming convention, e.g.:
    dataset12_SOL_Time_W0.25O0.5train.mat
    labels12_SOL_Time_W0.25O0.5train.mat
    """
    muscle = muscle.upper()

    train_data_file = data_dir / f"dataset12_{muscle}_{window_label}train.mat"
    test_data_file = data_dir / f"dataset12_{muscle}_{window_label}test.mat"
    train_label_file = data_dir / f"labels12_{muscle}_{window_label}train.mat"
    test_label_file = data_dir / f"labels12_{muscle}_{window_label}test.mat"

    x_train = load_mat_variable(train_data_file, "datasets")
    x_test = load_mat_variable(test_data_file, "datasets")
    y_train = load_mat_variable(train_label_file, "labelss")
    y_test = load_mat_variable(test_label_file, "labelss")

    x_train = x_train.reshape((-1, n_channels, n_samples, 1)).astype("float32")
    x_test = x_test.reshape((-1, n_channels, n_samples, 1)).astype("float32")

    y_train = np.asarray(y_train).reshape(-1).astype(int)
    y_test = np.asarray(y_test).reshape(-1).astype(int)

    return x_train, x_test, y_train, y_test


def build_deep_convnet(
    n_classes: int = 4,
    n_channels: int = 64,
    n_samples: int = 512,
    dropout_rate: float = 0.25,
    learning_rate: float = 1e-4,
) -> Model:
    """Build a DeepConvNet-style CNN for EMG classification."""
    input_layer = Input(shape=(n_channels, n_samples, 1))

    x = Conv2D(25, (1, 5), kernel_constraint=max_norm(2.0, axis=(0, 1, 2)))(input_layer)
    x = Conv2D(25, (n_channels, 1), kernel_constraint=max_norm(2.0, axis=(0, 1, 2)))(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.9)(x)
    x = Activation("elu")(x)
    x = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(x)
    x = Dropout(dropout_rate)(x)

    for filters in [50, 100, 200]:
        x = Conv2D(filters, (1, 5), kernel_constraint=max_norm(2.0, axis=(0, 1, 2)))(x)
        x = BatchNormalization(epsilon=1e-5, momentum=0.9)(x)
        x = Activation("elu")(x)
        x = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(x)
        x = Dropout(dropout_rate)(x)

    x = Flatten()(x)
    output_layer = Dense(n_classes, kernel_constraint=max_norm(0.5), activation="softmax")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def posture_wise_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Compute accuracy separately for each posture class."""
    rows = []
    for class_idx, posture_name in enumerate(POSTURE_NAMES):
        mask = y_true == class_idx
        acc = accuracy_score(y_true[mask], y_pred[mask]) if np.any(mask) else np.nan
        rows.append(
            {
                "class_index": class_idx,
                "posture": posture_name,
                "n_samples": int(np.sum(mask)),
                "accuracy": acc,
            }
        )
    return pd.DataFrame(rows)


def plot_training_history(history: tf.keras.callbacks.History, output_path: Path) -> None:
    """Save training and validation accuracy plot."""
    plt.figure(figsize=(7, 5))
    plt.plot(history.history["accuracy"], label="Training accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    """Save confusion matrix plot."""
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(POSTURE_NAMES))
    plt.xticks(tick_marks, POSTURE_NAMES, rotation=45, ha="right")
    plt.yticks(tick_marks, POSTURE_NAMES)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def train_and_evaluate(args: argparse.Namespace) -> Dict[str, float]:
    """Run the full training and evaluation pipeline."""
    set_random_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train_raw, y_test_raw = load_emg_dataset(
        data_dir=data_dir,
        muscle=args.muscle,
        n_channels=args.n_channels,
        n_samples=args.n_samples,
        window_label=args.window_label,
    )

    # Original MATLAB labels are assumed to be 1, 2, 3, 4. Convert to 0, 1, 2, 3.
    y_train = to_categorical(y_train_raw - 1, num_classes=args.n_classes)
    y_test = to_categorical(y_test_raw - 1, num_classes=args.n_classes)

    model = build_deep_convnet(
        n_classes=args.n_classes,
        n_channels=args.n_channels,
        n_samples=args.n_samples,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        verbose=1,
    )

    probabilities = model.predict(x_test)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = np.argmax(y_test, axis=1)

    overall_accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    class_accuracy = posture_wise_accuracy(y_true, y_pred)

    prefix = args.muscle.upper()
    plot_training_history(history, output_dir / f"{prefix}_training_history.png")
    plot_confusion_matrix(cm, output_dir / f"{prefix}_confusion_matrix.png")

    class_accuracy.to_csv(output_dir / f"{prefix}_posture_wise_accuracy.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "muscle": prefix,
                "overall_accuracy": overall_accuracy,
                "n_train": x_train.shape[0],
                "n_test": x_test.shape[0],
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "dropout_rate": args.dropout_rate,
                "seed": args.seed,
            }
        ]
    )
    summary.to_csv(output_dir / f"{prefix}_summary.csv", index=False)

    print("\nFinal results")
    print("-------------")
    print(f"Muscle: {prefix}")
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    print("\nPosture-wise accuracy:")
    print(class_accuracy.to_string(index=False))

    return {"overall_accuracy": overall_accuracy}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify standing postures from SOL or FDB EMG using a CNN."
    )
    parser.add_argument("--muscle", type=str, default="SOL", choices=["SOL", "FDB", "sol", "fdb"])
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--window_label", type=str, default="Time_W0.25O0.5")
    parser.add_argument("--n_channels", type=int, default=64)
    parser.add_argument("--n_samples", type=int, default=512)
    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.25)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train_and_evaluate(parse_args())
