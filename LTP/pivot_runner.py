"""
Trains the Pivoter LTP
"""

import torch.optim as optim
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from pivot_arch import *
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import os
import wandb
from tqdm import tqdm
from torch.utils.data import Subset

# Initialize wandb

wandb.init(project="Enhance Simplex", config={
    "learning_rate": 0.001,
    "epochs": 300,
    "batch_size": 16,  # Update to match DataLoader batch size
    "hidden_dim": 128,  # Update to match model hidden dimension
    "input_dim": 3     # Update to match model input dimension
})

# Load bipartite graph dataset
graphs_path = "/home/ac/Learning_To_Pivot/bipartite_graph_pivots"
data = LPDataset(graphs_path)

indices = list(range(len(data)))
# Create DataLoader instances
train_indices, val_indices = train_test_split(indices, train_size=0.8, random_state=42)
train_data = Subset(data, train_indices)
val_data = Subset(data, val_indices)

train_loader = DataLoader(train_data, batch_size=100, shuffle=True, collate_fn=collate_fn, num_workers=16)
val_loader = DataLoader(val_data, batch_size=100, shuffle=True, collate_fn=collate_fn, num_workers=16)

torch.cuda.set_device(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = PivotGCN(input_dim=3, hidden_dim=128, output_dim=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.BCEWithLogitsLoss()
#Checkpoint1 ->  GCNConv(GCN -> Tanh), fc2 (linear -> sigmoid), fc_opt
checkpoint_path = "/home/ac/Learning_To_Pivot/checkpoint2.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['loss']
else:
    start_epoch = 0
    best_loss = float('inf')

num_epochs = 300
for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0
    for data in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        data = data.to(device)
        optimizer.zero_grad()
        out, optimal = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = 0

        for count, k in enumerate(data.y):
            if k[0] == k[1]:
                loss += 1 * criterion(out[count], k)
            else:
                loss += 2000 * criterion(out[count], k)
        
        for count, k in enumerate(data.is_optimal.float()):
            if k.item() == 0:
                loss += 2100 * F.binary_cross_entropy(optimal.squeeze(dim=1)[count], k)
                uncertainty_penalty = 150 * (1 - 4 * (optimal.squeeze(dim=1)[count] - 0.5).pow(2))
            else:
                loss += 400000 * F.binary_cross_entropy(optimal.squeeze(dim=1)[count], k)
                uncertainty_penalty = 20000 * (1 - 4 * (optimal.squeeze(dim=1)[count] - 0.5).pow(2))

            loss += uncertainty_penalty

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    
    avg_loss = total_loss / len(train_data)
    wandb.log({"Train Loss": avg_loss, "epoch": epoch})
    print(f"Epoch {epoch + 1}, Train Avg Loss: {avg_loss:.4f}")

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
            data = data.to(device)
            out, optimal = model(data.x, data.edge_index, data.edge_attr, data.batch)

            for count, k in enumerate(data.y):
                if k[0] == k[1]:
                    loss += 1 * criterion(out[count], k)
                else:
                    loss += 2000 * criterion(out[count], k)
            
            for count, k in enumerate(data.is_optimal.float()):
                if k.item() == 0:
                    loss += 2100 * F.binary_cross_entropy(optimal.squeeze(dim=1)[count], k)
                    uncertainty_penalty = 150 * (1 - 4 * (optimal.squeeze(dim=1)[count] - 0.5).pow(2))
                else:
                    loss += 400000 * F.binary_cross_entropy(optimal.squeeze(dim=1)[count], k)
                    uncertainty_penalty = 20000 * (1 - 4 * (optimal.squeeze(dim=1)[count] - 0.5).pow(2))

                loss += uncertainty_penalty
            
            val_loss += loss.item() * data.num_graphs

    avg_val_loss = val_loss / len(val_data)
    wandb.log({"Validation Loss": avg_val_loss, "epoch": epoch})
    print(f"Epoch {epoch + 1}, Validation Avg Loss: {avg_val_loss:.4f}")

    # Save checkpoint
    if avg_val_loss < best_loss or epoch < 10:
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss,
        }, checkpoint_path)