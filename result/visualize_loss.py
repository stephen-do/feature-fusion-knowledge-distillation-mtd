import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('loss.csv')

def normalize(arr):
    arr = np.array(arr)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


epochs = df['epoch']
val_loss_kd = normalize(df['val_loss_kd'])
val_loss_feature = normalize(df['val_loss_feature'])
val_loss_det = normalize(df['val_loss_det'])
train_loss_kd = normalize(df['train_loss_kd'])
train_loss_feature = normalize(df['train_loss_feature'])
train_loss_det = normalize(df['train_loss_det'])

plt.plot(epochs, val_loss_kd, label='Val KD Loss', linestyle='--', color='blue')
plt.plot(epochs, val_loss_feature, label='Val Feature Loss', linestyle='--', color='green')
plt.plot(epochs, val_loss_det, label='Val Detection Loss', linestyle='--', color='orange')
plt.plot(epochs, train_loss_kd, label='Train KD Loss', color='blue', linewidth=2.5)
plt.plot(epochs, train_loss_feature, label='Train Feature Loss', color='green', linewidth=2.5)
plt.plot(epochs, train_loss_det, label='Train Detection Loss', color='orange', linewidth=2.5)

plt.xlabel('Epoch')
plt.ylabel('Normalized Loss')
plt.title('Normalized Loss over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()
plt.savefig('loss_plot.png')