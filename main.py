import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ── Data (with small realistic noise) ────────────────────────────────
torch.manual_seed(42)  # for reproducibility
X = torch.linspace(0, 10, 100).unsqueeze(1)
y_true = 3 * X + 1
y = y_true + torch.randn_like(X) * 0.15   # small noise

# ── NEW: Train / Validation split ────────────────────────────────────
n_samples = len(X)
indices = torch.randperm(n_samples)          # random shuffle of 0..99

train_size = int(0.8 * n_samples)             # 80 examples for training
val_size   = n_samples - train_size           # 20 examples for validation

train_idx = indices[:train_size]
val_idx   = indices[train_size:]

X_train = X[train_idx]
y_train = y[train_idx]

X_val   = X[val_idx]
y_val   = y[val_idx]

print(f"Training samples: {len(X_train)}    Validation samples: {len(X_val)}")

# ── Model ─────────────────────────────────────────────────────────────
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 24)       # slightly larger hidden layer
        self.layer2 = nn.Linear(24, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ── Training loop – now uses train split + reports both losses ───────
epochs = 100
train_losses = []
val_losses   = []

for epoch in range(epochs):
    # ── Training step ────────────────────────────────
    model.train()                   # important habit (even if no dropout here)
    pred = model(X_train)
    loss = criterion(pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # ── Validation step (no gradients needed) ────────
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val)

    # Store for plotting later
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    # Print progress
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d}   train loss: {loss.item():.6f}   val loss: {val_loss.item():.6f}")

# ── Final evaluation on the same test points as before ───────────────
model.eval()
with torch.no_grad():
    x_test = torch.tensor([[4.0]])
    y_pred = model(x_test).item()
    print(f"\nPrediction at x = 4.0 : {y_pred:.4f}   (target = 13.0000)")

    x0  = torch.tensor([[0.0]])
    x10 = torch.tensor([[10.0]])
    y0  = model(x0).item()
    y10 = model(x10).item()

    slope     = (y10 - y0) / 10.0
    intercept = y0

    print(f"Effective slope     : {slope:.4f}   (target = 3.0000)")
    print(f"Effective intercept : {intercept:.4f}   (target = 1.0000)")

# ── NEW: Plot train vs validation loss ───────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train loss', color='blue', alpha=0.7)
plt.plot(val_losses,   label='Validation loss', color='orange', alpha=0.7)
plt.yscale('log')                     # log scale helps see early fast drops
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()