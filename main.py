import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ── Data (with small realistic noise) ────────────────────────────────
torch.manual_seed(42)  # for reproducibility
X = torch.linspace(0, 10, 100).unsqueeze(1)
y_true = 3 * X + 1
y = y_true + torch.randn_like(X) * 0.15   # small noise

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

# ── Training ──────────────────────────────────────────────────────────
epochs = 3000
for epoch in range(epochs):
    pred = model(X)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d}   Loss: {loss.item():.6f}")

# ── Evaluation ────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    # Test point
    x_test = torch.tensor([[4.0]])
    y_pred = model(x_test).item()
    print(f"\nPrediction at x = 4.0 : {y_pred:.4f}   (target = 13.0000)")

    # Estimate slope & intercept numerically
    x0  = torch.tensor([[0.0]])
    x10 = torch.tensor([[10.0]])
    y0  = model(x0).item()
    y10 = model(x10).item()

    slope     = (y10 - y0) / 10.0
    intercept = y0

    print(f"Effective slope     : {slope:.4f}   (target = 3.0000)")
    print(f"Effective intercept : {intercept:.4f}   (target = 1.0000)")

# ── Visualization ─────────────────────────────────────────────────────
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