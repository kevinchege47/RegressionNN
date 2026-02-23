import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ── Data (with small realistic noise) ────────────────────────────────
torch.manual_seed(42)
X = torch.linspace(0, 10, 1000).unsqueeze(1)
y_true = 3 * X + 1
y = y_true + torch.randn_like(X) * 0.15

# ── Train / Validation split ─────────────────────────────────────────
n_samples = len(X)
indices = torch.randperm(n_samples)
train_size = int(0.8 * n_samples)
val_size = n_samples - train_size

train_idx = indices[:train_size]
val_idx   = indices[train_size:]

X_train, y_train = X[train_idx], y[train_idx]
X_val,   y_val   = X[val_idx],   y[val_idx]

print(f"Dataset split → Train: {len(X_train)} samples | Val: {len(X_val)} samples\n")

# ── Model ─────────────────────────────────────────────────────────────
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 24)
        self.layer2 = nn.Linear(24, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01,weight_decay=1e-3)

# Checkpoint tracking
best_val_loss = float('inf')
best_epoch = 0
best_model_path = 'best_model.pt'

# ── Training loop with Early Stopping ─────────────────────────────────────────────────────
epochs = 1000
train_losses = []
val_losses = []

patience = 60              # stop if no improvement for 60 epochs
min_delta = 0.0001          # minimum improvement to count as better
counter = 0                 # how many epochs since last improvement

print("Starting training with early stopping...\n")

for epoch in range(epochs):
    # Training step
    model.train()
    pred = model(X_train)
    loss = criterion(pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation step
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    # Check for improvement & save best
    improved = False
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss.item()
        best_epoch = epoch
        torch.save(model.state_dict(), best_model_path)
        print(f"  New best @ epoch {epoch:4d}  ──  val loss: {val_loss:8.6f}")
        improved = True
        counter = 0                          # reset patience counter
    else:
        counter += 1

    # Progress print
    if (epoch + 1) % 200 == 0 or epoch == 0:
        print(f"Epoch {epoch:4d}/{epochs-1:4d}   train: {loss:8.6f}   val: {val_loss:8.6f}")

    # Early stopping check
    if counter >= patience:
        print(f"\nEarly stopping triggered at epoch {epoch}")
        print(f"  No improvement for {patience} epochs → stopping")
        break

# ── Summary after loop ends (early or normal) ────────────────────────
print("\n" + "═" * 70)
print(f"Training stopped at epoch {epoch}")
print(f"  Best validation MSE: {best_val_loss:.6f} at epoch {best_epoch}")
print(f"  → model saved to: {best_model_path}")
print("═" * 70 + "\n")

# ── Load best model & evaluate ────────────────────────────────────────
model.load_state_dict(torch.load(best_model_path))
print("Evaluating best model...\n")

model.eval()
with torch.no_grad():
    # Test points
    test_points = torch.tensor([[0.0], [4.0], [10.0]])
    preds = model(test_points).squeeze().tolist()

    print("Test predictions:")
    for x_val, pred in zip([0.0, 4.0, 10.0], preds):
        target = 3 * x_val + 1
        err = abs(pred - target)
        print(f"  x = {x_val:4.1f}  →  pred = {pred:8.4f}   target = {target:6.1f}   error = {err:.4f}")

    # Slope & intercept
    y0, y10 = preds[0], preds[2]
    slope = (y10 - y0) / 10.0
    intercept = y0

    print("\nLearned linear behavior:")
    print(f"  Slope     = {slope:8.4f}   (target 3.0000)")
    print(f"  Intercept = {intercept:8.4f}   (target 1.0000)")
    print("═" * 70)

# ── Plot losses ───────────────────────────────────────────────────────
plt.figure(figsize=(9, 6))
plt.scatter(X.numpy(), y.numpy(), s=40, alpha=0.6, label="Noisy data")
plt.plot(X.numpy(), model(X).detach().numpy(), color="red", lw=2.2, label="Learned fit")
plt.plot(X.numpy(), y_true.numpy(), "--", color="gray", label="True: y = 3x + 1")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Neural Network Fit to Linear Relationship")
plt.legend(frameon=True, shadow=True)
plt.grid(True, alpha=0.3)

plt.show()