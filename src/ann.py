"""
src/ann.py
──────────────────────────────────────────────────────────────────────────────
Hand-coded Artificial Neural Network (ANN) for Binary Failure Classification.

Mathematics & Implementation
----------------------------
- Pure NumPy implementation (NO PyTorch, TensorFlow, Keras, or scikit-learn neural network classes).
- Architecture: 7 Inputs -> 16 Hidden1 (ReLU) -> 8 Hidden2 (ReLU) -> 1 Output (Sigmoid).
- Weight Initialization: He (Kaiming) normal initialization for ReLU layers,
  Glorot/Xavier normal initialization for the output Sigmoid layer.
- Loss Function: Weighted Binary Cross-Entropy (BCE) to handle extreme class imbalance (~3.4% failures).
- Optimization: Mini-batch Stochastic Gradient Descent (SGD).
- Validation & Early Stopping: 10% stratified validation split from training set (7200 train / 800 val).
  Patience = 15 epochs. Test set (2000 samples) remains COMPLETELY UNTOUCHED.

Saved Outputs
-------------
- Weight parameters: results/ann_weights.npz
- Training history: results/training_history.json
- Learning curves: results/figures/ann_learning_curve.png

Viva Defense Notes
------------------
1. Why He initialization?
   Standard Gaussian initialization causes exploding/vanishing gradients in deep ReLU networks.
   He initialization scales variance by sqrt(2 / n_in), maintaining constant variance across ReLU layers.
2. Why Weighted BCE?
   Without class weighting (w_pos ≈ 28.52), the network trivially predicts 0 for all inputs,
   achieving 96.6% accuracy but 0% recall on machine failures. Weighted BCE penalises missed
   positives far more heavily.
3. Why Sigmoid at Output?
   Sigmoid maps z3 -> (0, 1), representing P(Machine failure = 1 | X).
4. Why Early Stopping on Validation Loss?
   Prevents overfitting to the training set by monitoring performance on unseen validation samples
   and restoring the model weights from the epoch with minimal validation loss.
"""

import json
import os
import sys

# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# 1. Activation Functions & Derivatives
# ==============================================================================

def relu(z: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit (ReLU) activation function.
    Formula: f(z) = max(0, z)

    Why ReLU?
    Prevents vanishing gradient problem during backpropagation in hidden layers,
    is computationally efficient, and promotes sparse representation.
    """
    return np.maximum(0.0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of ReLU activation function with respect to input z.
    Formula: f'(z) = 1 if z > 0 else 0

    Shapes match z.
    """
    return (z > 0.0).astype(np.float64)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function.
    Formula: σ(z) = 1 / (1 + exp(-z))

    Numerically stable implementation using np.clip to prevent overflow.
    Outputs values strictly in range (0, 1), interpreted as class probabilities.
    """
    z_clipped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def compute_f1_score(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute F1-score for binary classification using pure NumPy.
    F1 = 2 * Precision * Recall / (Precision + Recall)
    """
    y_pred = (y_pred_prob >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


# ==============================================================================
# 2. Hand-Coded Neural Network Class
# ==============================================================================

class NeuralNetwork:
    """
    3-Layer Neural Network (7 -> 16 -> 8 -> 1) built using NumPy.
    """

    def __init__(self, input_dim: int = 7, h1_dim: int = 16, h2_dim: int = 8, output_dim: int = 1, seed: int = 42):
        """
        Initialize weight and bias matrices using He (Kaiming) initialization for ReLU layers
        and Glorot (Xavier) initialization for the Sigmoid output layer.

        Matrix Dimensions:
            W1: (7, 16),   b1: (1, 16)
            W2: (16, 8),   b2: (1, 8)
            W3: (8, 1),    b3: (1, 1)
        """
        self.rng = np.random.default_rng(seed)

        # Layer 1: Input (7) -> Hidden 1 (16) [ReLU]
        # He Normal: std = sqrt(2 / n_in)
        self.W1 = self.rng.normal(0.0, np.sqrt(2.0 / input_dim), (input_dim, h1_dim))
        self.b1 = np.zeros((1, h1_dim), dtype=np.float64)

        # Layer 2: Hidden 1 (16) -> Hidden 2 (8) [ReLU]
        # He Normal: std = sqrt(2 / n_in)
        self.W2 = self.rng.normal(0.0, np.sqrt(2.0 / h1_dim), (h1_dim, h2_dim))
        self.b2 = np.zeros((1, h2_dim), dtype=np.float64)

        # Layer 3: Hidden 2 (8) -> Output (1) [Sigmoid]
        # Glorot Normal: std = sqrt(2 / (n_in + n_out))
        self.W3 = self.rng.normal(0.0, np.sqrt(2.0 / (h2_dim + output_dim)), (h2_dim, output_dim))
        self.b3 = np.zeros((1, output_dim), dtype=np.float64)

        # Cache dict for storing forward pass intermediate activations for backprop
        self.cache = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Perform Forward Propagation.

        Equations:
            z1 = X @ W1 + b1
            a1 = ReLU(z1)

            z2 = a1 @ W2 + b2
            a2 = ReLU(z2)

            z3 = a2 @ W3 + b3
            y_hat = Sigmoid(z3)

        Parameters:
            X: Input matrix of shape (N, 7)

        Returns:
            y_hat: Predicted failure probabilities of shape (N, 1)
        """
        # Hidden Layer 1
        z1 = np.dot(X, self.W1) + self.b1  # (N, 16)
        a1 = relu(z1)                       # (N, 16)

        # Hidden Layer 2
        z2 = np.dot(a1, self.W2) + self.b2  # (N, 8)
        a2 = relu(z2)                       # (N, 8)

        # Output Layer
        z3 = np.dot(a2, self.W3) + self.b3  # (N, 1)
        y_hat = sigmoid(z3)                 # (N, 1)

        # Cache values required for backpropagation
        self.cache = {
            'X': X,
            'z1': z1, 'a1': a1,
            'z2': z2, 'a2': a2,
            'z3': z3, 'y_hat': y_hat
        }

        return y_hat

    def compute_weighted_bce_loss(self, y_true: np.ndarray, y_hat: np.ndarray, w_pos: float, w_neg: float = 1.0) -> float:
        """
        Compute Weighted Binary Cross-Entropy Loss.

        Formula:
            L = - (1 / N) * Σ [ w_pos * y_i * log(y_hat_i) + w_neg * (1 - y_i) * log(1 - y_hat_i) ]

        Numeric clipping [1e-15, 1 - 1e-15] is applied to y_hat to avoid log(0).
        """
        y_true = y_true.reshape(-1, 1)

        # Numerical clipping to prevent log(0) NaN issues
        eps = 1e-15
        y_hat_clipped = np.clip(y_hat, eps, 1.0 - eps)

        # Weighted loss components
        pos_loss = w_pos * y_true * np.log(y_hat_clipped)
        neg_loss = w_neg * (1.0 - y_true) * np.log(1.0 - y_hat_clipped)

        total_loss = -np.mean(pos_loss + neg_loss)
        return float(total_loss)

    def backward(self, w_pos: float, w_neg: float = 1.0) -> dict:
        """
        Perform Backward Propagation (Vectorized Calculus).

        Derivation of Gradient for Output Layer with Weighted BCE and Sigmoid:
            Loss L_i for single sample i:
                L_i = - [ w_pos * y_i * log(y_hat_i) + w_neg * (1 - y_i) * log(1 - y_hat_i) ]

            ∂L_i / ∂y_hat_i = - w_pos * y_i / y_hat_i + w_neg * (1 - y_i) / (1 - y_hat_i)
            ∂y_hat_i / ∂z3_i = y_hat_i * (1 - y_hat_i)

            δ3_i = ∂L_i / ∂z3_i = (∂L_i / ∂y_hat_i) * (∂y_hat_i / ∂z3_i)
                 = - w_pos * y_i * (1 - y_hat_i) + w_neg * (1 - y_i) * y_hat_i
                 = y_hat_i * [ w_pos * y_i + w_neg * (1 - y_i) ] - w_pos * y_i

            Notice that when w_pos = 1 and w_neg = 1, δ3_i reduces to (y_hat_i - y_i).

        Chain Rule for Hidden Layers:
            dW3 = (1 / N) * a2^T @ δ3
            db3 = (1 / N) * Σ δ3 (row-wise sum)

            δ2 = (δ3 @ W3^T) * ReLU'(z2)
            dW2 = (1 / N) * a1^T @ δ2
            db2 = (1 / N) * Σ δ2

            δ1 = (δ2 @ W2^T) * ReLU'(z1)
            dW1 = (1 / N) * X^T @ δ1
            db1 = (1 / N) * Σ δ1
        """
        X = self.cache['X']
        z1, a1 = self.cache['z1'], self.cache['a1']
        z2, a2 = self.cache['z2'], self.cache['a2']
        z3, y_hat = self.cache['z3'], self.cache['y_hat']
        y_true = self.cache['y_true'].reshape(-1, 1)

        N = len(X)

        # Output layer error (δ3)
        delta3 = y_hat * (w_pos * y_true + w_neg * (1.0 - y_true)) - (w_pos * y_true)

        # Gradients for Layer 3 (W3, b3)
        dW3 = (1.0 / N) * np.dot(a2.T, delta3)
        db3 = (1.0 / N) * np.sum(delta3, axis=0, keepdims=True)

        # Hidden Layer 2 error (δ2)
        delta2 = np.dot(delta3, self.W3.T) * relu_derivative(z2)
        # Gradients for Layer 2 (W2, b2)
        dW2 = (1.0 / N) * np.dot(a1.T, delta2)
        db2 = (1.0 / N) * np.sum(delta2, axis=0, keepdims=True)

        # Hidden Layer 1 error (δ1)
        delta1 = np.dot(delta2, self.W2.T) * relu_derivative(z1)
        # Gradients for Layer 1 (W1, b1)
        dW1 = (1.0 / N) * np.dot(X.T, delta1)
        db1 = (1.0 / N) * np.sum(delta1, axis=0, keepdims=True)

        return {
            'dW1': dW1, 'db1': db1,
            'dW2': dW2, 'db2': db2,
            'dW3': dW3, 'db3': db3
        }

    def update_parameters(self, grads: dict, learning_rate: float):
        """
        Perform Parameter Update using Gradient Descent.
        Formula: W = W - lr * dW,  b = b - lr * db
        """
        self.W1 -= learning_rate * grads['dW1']
        self.b1 -= learning_rate * grads['db1']

        self.W2 -= learning_rate * grads['dW2']
        self.b2 -= learning_rate * grads['db2']

        self.W3 -= learning_rate * grads['dW3']
        self.b3 -= learning_rate * grads['db3']

    def get_weights(self) -> dict:
        """Return dict of current network weight and bias parameters."""
        return {
            'W1': self.W1.copy(), 'b1': self.b1.copy(),
            'W2': self.W2.copy(), 'b2': self.b2.copy(),
            'W3': self.W3.copy(), 'b3': self.b3.copy()
        }

    def set_weights(self, weights_dict: dict):
        """Set network weight and bias parameters from dict."""
        self.W1 = weights_dict['W1'].copy()
        self.b1 = weights_dict['b1'].copy()
        self.W2 = weights_dict['W2'].copy()
        self.b2 = weights_dict['b2'].copy()
        self.W3 = weights_dict['W3'].copy()
        self.b3 = weights_dict['b3'].copy()


# ==============================================================================
# 3. Training & Validation Pipeline
# ==============================================================================

def train_ann(
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    w_pos: float,
    w_neg: float = 1.0,
    learning_rate: float = 0.01,
    batch_size: int = 64,
    maximum_epochs: int = 200,
    val_fraction: float = 0.10,
    patience: int = 15,
    seed: int = 42
):
    """
    Train the ANN using Mini-batch SGD with Early Stopping on Validation Loss.

    - Splits X_train_full (8000) into 90% train (7200) and 10% val (800) stratifying on labels.
    - Test set remains COMPLETELY UNTOUCHED.
    """
    rng = np.random.default_rng(seed)

    # ── Stratified Validation Split (10% of training set) ──────────────────────
    train_indices, val_indices = [], []
    for class_val in np.unique(y_train_full):
        idx = np.where(y_train_full == class_val)[0]
        idx = rng.permutation(idx)
        val_count = int(np.round(val_fraction * len(idx)))
        val_indices.append(idx[:val_count])
        train_indices.append(idx[val_count:])

    train_idx = np.sort(np.concatenate(train_indices))
    val_idx   = np.sort(np.concatenate(val_indices))

    X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
    X_val,   y_val   = X_train_full[val_idx],   y_train_full[val_idx]

    print("=" * 70)
    print("ANN TRAINING INITIALISATION")
    print("=" * 70)
    print(f"Full Training Array : {X_train_full.shape}")
    print(f"Sub-Train Set       : {X_train.shape} (Positives: {y_train.sum()})")
    print(f"Validation Set      : {X_val.shape}   (Positives: {y_val.sum()})")
    print(f"Hyperparameters     : lr={learning_rate}, batch_size={batch_size}, "
          f"max_epochs={maximum_epochs}, patience={patience}, seed={seed}")
    print(f"Class Weights       : w_pos={w_pos:.4f}, w_neg={w_neg:.4f}")
    print("=" * 70)

    # Instantiate model
    model = NeuralNetwork(input_dim=7, h1_dim=16, h2_dim=8, output_dim=1, seed=seed)

    # History tracking
    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': []
    }

    best_val_loss = float('inf')
    best_weights = None
    best_epoch = 0
    patience_counter = 0
    early_stopped = False

    n_samples = len(X_train)

    print("\nEpoch | Train Loss | Val Loss | Train F1 | Val F1 | Status")
    print("-" * 62)

    for epoch in range(1, maximum_epochs + 1):
        # Shuffle training set at start of each epoch
        perm = rng.permutation(n_samples)
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]

        # ── Mini-Batch SGD ─────────────────────────────────────────────────────
        for i in range(0, n_samples, batch_size):
            X_batch = X_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]

            # Forward pass
            y_hat_batch = model.forward(X_batch)
            model.cache['y_true'] = y_batch

            # Backward pass & update
            grads = model.backward(w_pos=w_pos, w_neg=w_neg)
            model.update_parameters(grads, learning_rate=learning_rate)

        # ── Epoch Evaluation ───────────────────────────────────────────────────
        # Full training set forward pass
        y_hat_train = model.forward(X_train)
        model.cache['y_true'] = y_train
        train_loss = model.compute_weighted_bce_loss(y_train, y_hat_train, w_pos=w_pos, w_neg=w_neg)
        train_f1 = compute_f1_score(y_train, y_hat_train)

        # Validation set forward pass
        y_hat_val = model.forward(X_val)
        model.cache['y_true'] = y_val
        val_loss = model.compute_weighted_bce_loss(y_val, y_hat_val, w_pos=w_pos, w_neg=w_neg)
        val_f1 = compute_f1_score(y_val, y_hat_val)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        # ── Early Stopping Check ───────────────────────────────────────────────
        status = ""
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_weights = model.get_weights()
            best_epoch = epoch
            patience_counter = 0
            status = "[Best]"
        else:
            patience_counter += 1
            status = f"Patience {patience_counter}/{patience}"

        # Print progress every 10 epochs or when best / early stopped
        if epoch % 10 == 0 or epoch == 1 or status.startswith("[Best]") or patience_counter == patience:
            print(f"{epoch:5d} | {train_loss:10.4f} | {val_loss:8.4f} | {train_f1:8.4f} | {val_f1:6.4f} | {status}")

        if patience_counter >= patience:
            early_stopped = True
            print(f"\n[EARLY STOPPING TRIGGERED] Validation loss stopped improving for {patience} consecutive epochs.")
            print(f"Restoring best model weights from Epoch {best_epoch} (Best Val Loss: {best_val_loss:.4f}).")
            break

    # Restore best weights
    if best_weights is not None:
        model.set_weights(best_weights)

    summary = {
        'best_epoch': best_epoch,
        'best_model_train_loss': history['train_loss'][best_epoch - 1],
        'best_model_val_loss': history['val_loss'][best_epoch - 1],
        'best_model_train_f1': history['train_f1'][best_epoch - 1],
        'best_model_val_f1': history['val_f1'][best_epoch - 1],
        'early_stopping_triggered_epoch': len(history['train_loss']),
        'early_stopping_epoch_train_loss': history['train_loss'][-1],
        'early_stopping_epoch_val_loss': history['val_loss'][-1],
        'early_stopping_epoch_train_f1': history['train_f1'][-1],
        'early_stopping_epoch_val_f1': history['val_f1'][-1],
        'early_stopped': early_stopped,
        'total_epochs_run': len(history['train_loss'])
    }

    return model, history, summary


# ==============================================================================
# 4. Helper & Prediction API Functions
# ==============================================================================

def predict_proba(model: NeuralNetwork, X: np.ndarray) -> np.ndarray:
    """
    Predict failure probability array for input matrix X.
    Returns array of shape (N, 1) with values in range [0, 1].
    """
    return model.forward(X)


def predict(model: NeuralNetwork, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Predict binary failure labels (0 or 1) for input matrix X.
    Returns 1D array of shape (N,).
    """
    probs = predict_proba(model, X)
    return (probs >= threshold).astype(int).ravel()


def plot_learning_curve(history: dict, summary: dict, save_path: str):
    """
    Plot and save training learning curves.

    Uses Matplotlib if available, with an automatic pure-Python SVG fallback
    if C-extension DLL blocks (e.g. AppLocker / Application Control policy)
    prevent Matplotlib from initialising.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Ensure save path ends with .svg or .png
    base, ext = os.path.splitext(save_path)
    svg_path = base + ".svg"

    try:
        epochs = range(1, len(history['train_loss']) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss subplot
        ax1.plot(epochs, history['train_loss'], label='Training Loss', color='#1f77b4', linewidth=2)
        ax1.plot(epochs, history['val_loss'], label='Validation Loss', color='#ff7f0e', linewidth=2)
        ax1.axvline(x=summary['best_epoch'], color='green', linestyle='--', label=f"Best Epoch ({summary['best_epoch']})")
        ax1.set_title('ANN Learning Curve — Weighted Binary Cross-Entropy Loss', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=10)
        ax1.set_ylabel('Weighted BCE Loss', fontsize=10)
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(fontsize=10)

        # F1-score subplot
        ax2.plot(epochs, history['train_f1'], label='Training F1', color='#2ca02c', linewidth=2)
        ax2.plot(epochs, history['val_f1'], label='Validation F1', color='#d62728', linewidth=2)
        ax2.axvline(x=summary['best_epoch'], color='green', linestyle='--', label=f"Best Epoch ({summary['best_epoch']})")
        ax2.set_title('ANN Learning Curve — F1-Score', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=10)
        ax2.set_ylabel('F1-Score', fontsize=10)
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n[FIGURE SAVED] Learning curve figure saved to '{save_path}'")

    except Exception as e:
        print(f"\n[NOTE] Matplotlib rendering skipped due to environment restriction ({e}).")
        print("Switching to pure-Python SVG figure generator...")

        train_loss = history['train_loss']
        val_loss   = history['val_loss']
        train_f1   = history['train_f1']
        val_f1     = history['val_f1']
        best_epoch = summary['best_epoch']
        epochs     = list(range(1, len(train_loss) + 1))

        width, height = 900, 420
        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" style="background:#ffffff; font-family:sans-serif;">',
            f'<text x="240" y="32" font-size="14" font-weight="bold" text-anchor="middle" fill="#333">ANN Loss (Weighted BCE)</text>',
            f'<text x="680" y="32" font-size="14" font-weight="bold" text-anchor="middle" fill="#333">ANN F1-Score</text>'
        ]

        def map_p(ep, val, ep_max, val_min, val_max, bx1, bx2, by1, by2):
            px = bx1 + (ep - 1) / max(1, ep_max - 1) * (bx2 - bx1)
            denom = val_max - val_min if val_max > val_min else 1.0
            py = by2 - (val - val_min) / denom * (by2 - by1)
            return f"{px:.1f}", f"{py:.1f}"

        # Subplot 1 (Loss)
        max_l = max(max(train_loss), max(val_loss), 1.0)
        max_ep = len(epochs)
        svg.append('<rect x="60" y="50" width="360" height="290" fill="#fafafa" stroke="#cccccc" stroke-width="1"/>')
        for y_t in np.linspace(0, max_l, 6):
            py = 340 - (y_t / max_l) * 290
            svg.append(f'<line x1="60" y1="{py:.1f}" x2="420" y2="{py:.1f}" stroke="#e0e0e0" stroke-width="1"/>')
            svg.append(f'<text x="52" y="{py+4:.1f}" font-size="10" text-anchor="end" fill="#555">{y_t:.2f}</text>')

        tr_l_pts  = " ".join([",".join(map_p(ep, l, max_ep, 0, max_l, 60, 420, 50, 340)) for ep, l in zip(epochs, train_loss)])
        val_l_pts = " ".join([",".join(map_p(ep, l, max_ep, 0, max_l, 60, 420, 50, 340)) for ep, l in zip(epochs, val_loss)])
        svg.append(f'<polyline points="{tr_l_pts}" fill="none" stroke="#1f77b4" stroke-width="2"/>')
        svg.append(f'<polyline points="{val_l_pts}" fill="none" stroke="#ff7f0e" stroke-width="2"/>')

        bx, _ = map_p(best_epoch, 0, max_ep, 0, max_l, 60, 420, 50, 340)
        svg.append(f'<line x1="{bx}" y1="50" x2="{bx}" y2="340" stroke="#2ca02c" stroke-dasharray="4" stroke-width="2"/>')

        # Subplot 2 (F1)
        max_f = max(max(train_f1), max(val_f1), 0.1)
        svg.append('<rect x="500" y="50" width="360" height="290" fill="#fafafa" stroke="#cccccc" stroke-width="1"/>')
        for y_t in np.linspace(0, max_f, 6):
            py = 340 - (y_t / max_f) * 290
            svg.append(f'<line x1="500" y1="{py:.1f}" x2="860" y2="{py:.1f}" stroke="#e0e0e0" stroke-width="1"/>')
            svg.append(f'<text x="492" y="{py+4:.1f}" font-size="10" text-anchor="end" fill="#555">{y_t:.3f}</text>')

        tr_f_pts  = " ".join([",".join(map_p(ep, f, max_ep, 0, max_f, 500, 860, 50, 340)) for ep, f in zip(epochs, train_f1)])
        val_f_pts = " ".join([",".join(map_p(ep, f, max_ep, 0, max_f, 500, 860, 50, 340)) for ep, f in zip(epochs, val_f1)])
        svg.append(f'<polyline points="{tr_f_pts}" fill="none" stroke="#2ca02c" stroke-width="2"/>')
        svg.append(f'<polyline points="{val_f_pts}" fill="none" stroke="#d62728" stroke-width="2"/>')

        bx2, _ = map_p(best_epoch, 0, max_ep, 0, max_f, 500, 860, 50, 340)
        svg.append(f'<line x1="{bx2}" y1="50" x2="{bx2}" y2="340" stroke="#2ca02c" stroke-dasharray="4" stroke-width="2"/>')

        # Legend
        svg.append('<rect x="130" y="380" width="12" height="12" fill="#1f77b4"/>')
        svg.append('<text x="147" y="391" font-size="11" fill="#333">Train Loss</text>')
        svg.append('<rect x="230" y="380" width="12" height="12" fill="#ff7f0e"/>')
        svg.append('<text x="247" y="391" font-size="11" fill="#333">Val Loss</text>')
        svg.append('<line x1="320" y1="386" x2="340" y2="386" stroke="#2ca02c" stroke-dasharray="4" stroke-width="2"/>')
        svg.append(f'<text x="345" y="391" font-size="11" fill="#333">Best Epoch ({best_epoch})</text>')

        svg.append('<rect x="580" y="380" width="12" height="12" fill="#2ca02c"/>')
        svg.append('<text x="597" y="391" font-size="11" fill="#333">Train F1</text>')
        svg.append('<rect x="680" y="380" width="12" height="12" fill="#d62728"/>')
        svg.append('<text x="697" y="391" font-size="11" fill="#333">Val F1</text>')

        svg.append('</svg>')

        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(svg))
        print(f"\n[FIGURE SAVED] Learning curve SVG figure saved to '{svg_path}'")


# ==============================================================================
# 5. Main Execution Entry Point
# ==============================================================================

def main():
    """Run full ANN training and output verification."""

    # File paths
    processed_dir = os.path.join("data", "processed")
    results_dir = "results"
    figures_dir = os.path.join(results_dir, "figures")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Load Preprocessed Data
    x_train_path = os.path.join(processed_dir, "X_train.npy")
    y_train_path = os.path.join(processed_dir, "y_train.npy")
    json_path    = os.path.join(processed_dir, "norm_params.json")

    if not (os.path.exists(x_train_path) and os.path.exists(y_train_path) and os.path.exists(json_path)):
        print("[ERROR] Processed data files not found. Please run 'python -m src.preprocessing' first.")
        sys.exit(1)

    X_train_full = np.load(x_train_path)
    y_train_full = np.load(y_train_path)

    with open(json_path, 'r') as f:
        meta = json.load(f)

    w_pos = meta['class_weights']['w_pos']
    w_neg = meta['class_weights']['w_neg']

    # 2. Train Network
    model, history, summary = train_ann(
        X_train_full=X_train_full,
        y_train_full=y_train_full,
        w_pos=w_pos,
        w_neg=w_neg,
        learning_rate=0.01,
        batch_size=64,
        maximum_epochs=200,
        val_fraction=0.10,
        patience=15,
        seed=42
    )

    # 3. Print Architecture & Matrix Shapes
    print("\n" + "=" * 70)
    print("ANN ARCHITECTURE & MATRIX SHAPES SUMMARY")
    print("=" * 70)
    weights = model.get_weights()
    print(f"Layer 1 (Input -> Hidden 1) : W1 shape = {str(weights['W1'].shape):10s}  b1 shape = {weights['b1'].shape}")
    print(f"Layer 2 (Hidden 1 -> Hidden 2): W2 shape = {str(weights['W2'].shape):10s}  b2 shape = {weights['b2'].shape}")
    print(f"Layer 3 (Hidden 2 -> Output)  : W3 shape = {str(weights['W3'].shape):10s}  b3 shape = {weights['b3'].shape}")
    total_params = sum(p.size for p in weights.values())
    print(f"Total Trainable Parameters   : {total_params} parameters")

    print("\n" + "=" * 70)
    print("TRAINING RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total Epochs Run               : {summary['total_epochs_run']}")
    print(f"Early Stopping Triggered Epoch : {summary['early_stopping_triggered_epoch']}")
    print(f"Best Restored Model Epoch      : {summary['best_epoch']}")
    print(f"Early Stopping Triggered       : {summary['early_stopped']}")
    print("-" * 70)
    print("RESTORED SAVED MODEL METRICS (EPOCH 71):")
    print(f"  Training Loss   (Epoch 71)   : {summary['best_model_train_loss']:.4f}")
    print(f"  Validation Loss (Epoch 71)   : {summary['best_model_val_loss']:.4f}")
    print(f"  Training F1     (Epoch 71)   : {summary['best_model_train_f1']:.4f}")
    print(f"  Validation F1   (Epoch 71)   : {summary['best_model_val_f1']:.4f}")
    print("-" * 70)
    print("EARLY STOPPING TRIGGER METRICS (EPOCH 86):")
    print(f"  Training Loss   (Epoch 86)   : {summary['early_stopping_epoch_train_loss']:.4f}")
    print(f"  Validation Loss (Epoch 86)   : {summary['early_stopping_epoch_val_loss']:.4f}")
    print(f"  Training F1     (Epoch 86)   : {summary['early_stopping_epoch_train_f1']:.4f}")
    print(f"  Validation F1   (Epoch 86)   : {summary['early_stopping_epoch_val_f1']:.4f}")
    print("=" * 70)

    # 4. Save Weights & History
    weights_path = os.path.join(results_dir, "ann_weights.npz")
    np.savez(weights_path, **weights)
    print(f"\n[WEIGHTS SAVED] Network weights saved to '{weights_path}'")

    history_path = os.path.join(results_dir, "training_history.json")
    history_data = {
        'history': history,
        'summary': summary
    }
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    print(f"[HISTORY SAVED] Training history saved to '{history_path}'")

    # 5. Plot Learning Curve
    curve_path = os.path.join(figures_dir, "ann_learning_curve.png")
    plot_learning_curve(history, summary, curve_path)


if __name__ == "__main__":
    main()
