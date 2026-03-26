import re
import matplotlib.pyplot as plt

exp_name = "EXP_26_3_3"
log_file = f"results/{exp_name}/training.log"

epochs = []
train_losses = []
val_losses = []
train_pck = []
val_pck = []
train_aed = []
val_aed = []
lr = []

pattern = re.compile(
    r"Epoch\s+(\d+).*?"
    r"train_loss=([0-9.eE+-]+).*?"
    r"train_metrics=\{[^}]*'PCKHeatmaps':\s*'([0-9.eE+-]+)'.*?'AEDHeatmaps':\s*'([0-9.eE+-]+)'[^}]*\}.*?"
    r"val_loss=([0-9.eE+-]+).*?"
    r"val_metrics=\{[^}]*'PCKHeatmaps':\s*'([0-9.eE+-]+)'.*?'AEDHeatmaps':\s*'([0-9.eE+-]+)'[^}]*\}.*?"
    r"lr=([0-9.eE+-]+)"
)

with open(log_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            epochs.append(int(match.group(1)))
            train_losses.append(float(match.group(2)))
            train_pck.append(float(match.group(3)))
            train_aed.append(float(match.group(4)))
            val_losses.append(float(match.group(5)))
            val_pck.append(float(match.group(6)))
            val_aed.append(float(match.group(7)))
            lr.append(float(match.group(8)))

# ===== Graf 1: Loss =====
plt.figure()
plt.plot(epochs, train_losses, label="Train loss")
plt.plot(epochs, val_losses, label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xlim(10, epochs[-1])
plt.ylim(0, 0.02)
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"results/{exp_name}/Loss.png")
plt.show()

# ===== Graf 2: PCK =====
plt.figure()
plt.plot(epochs, train_pck, label="Train PCK")
plt.plot(epochs, val_pck, label="Val PCK")
plt.xlabel("Epoch")
plt.ylabel("PCK")
plt.xlim(0, epochs[-1])
plt.title("Training vs Validation PCK")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"results/{exp_name}/PCK.png")
plt.show()

# ===== Graf 3: AED =====
plt.figure()
plt.plot(epochs, train_aed, label="Train AED")
plt.plot(epochs, val_aed, label="Val AED")
plt.xlabel("Epoch")
plt.ylabel("AED")
plt.xlim(0, epochs[-1])
plt.title("Training vs Validation AED")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"results/{exp_name}/AED.png")
plt.show()

# ===== Graf 4: lr =====
plt.figure()
plt.plot(epochs, lr, label="lr")
plt.xlabel("Epoch")
plt.ylabel("lr")
plt.xlim(0, epochs[-1])
plt.title("Learning rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"results/{exp_name}/lr.png")
plt.show()