"""
Train the slack SIB model.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree  # Importing required utilities
from arch import *
import wandb  # Import wandb

# Initialize wandb
wandb.init(project="Enhance Simplex", config={
    "learning_rate": 0.001,
    "epochs": 300,
    "batch_size": 6,
    "hidden_dim": 62,
    "input_dim": 2
})

# Set up dataset and dataloader
graphs_path = "/home/ac/Learning_To_Pivot/bipartie_graph2_replicate"
train_dataset = LPDataset(graphs_path, 0, 700)
val_dataset = LPDataset(graphs_path, 700, 800)
test_dataset = LPDataset(graphs_path, 800, 1000)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# Define model, optimizer, and loss function
input_dim = 2  # You can adjust based on your graph features
hidden_dim = 64  # You can adjust the hidden dimension
model = GCN(input_dim, hidden_dim)
#model.apply(initialize_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()
criterion_variable = nn.BCELoss()
criterion_slack = nn.BCELoss()

# Check if checkpoint exists
checkpoint_path = "/home/ac/Learning_To_Pivot/checkpoint3.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['loss']
else:
    start_epoch = 0
    best_loss = float('inf')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Transfer optimizer state to the correct device
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

# Training loop
num_epochs = 300
for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0
    slack_weight_1 = torch.tensor(1.5, device=device)
    slack_weight_0 = torch.tensor(1, device=device)
    alpha=4
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        #output = output.view(-1, 1)
        #print(output[0:500])
        #if not output.requires_grad:
            #print("Output does not require gradients")
        
        collected_results = []
        target_result = []
        index1 = 0
        index2 = 0
        for _ in range(data.num_graphs):
            index1 += 1200
            index2 += 1200
            collected = output[index1:index1 + 700]
            target = data.y.float()[index2:index2+700]
            collected_results.extend(collected)
            target_result.extend(target)
            index1 += 700
            index2 += 700

        result = torch.cat(collected_results, dim=0)
        label = torch.tensor([t.item() for t in target_result], device='cuda:0')
        slack_weights = torch.where(label == 1, slack_weight_1, slack_weight_0)
        #regularization = torch.mean(torch.where((result <= 0.2),
                                #(result - 0.5) ** 2,
                                #torch.zeros_like(result)))
        loss = (criterion(result, label) * slack_weights).mean() #+ alpha * regularization 
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    
    avg_loss = total_loss / len(train_dataset)
    wandb.log({"Train Loss": avg_loss, "epoch": epoch})
    print(f"Epoch {epoch + 1}, Train Avg Loss: {avg_loss:.4f}")

    # Validation loop
    model.eval()
    val_loss = 0
    swap = False
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            output = model(data)
            #output = output.view(-1, 1)
            #print(output[0:100])
            
            collected_results = []
            target_result = []
            index1 = 0
            index2 = 0
            for _ in range(data.num_graphs):
                index1 += 1200
                index2 += 1200
                collected = output[index1:index1 + 700]
                target = data.y.float()[index2:index2+700]
                collected_results.extend(collected)
                target_result.extend(target)
                index1 += 700
                index2 += 700
            
            print(collected_results[:700])
            result = torch.cat(collected_results, dim=0)
            label = torch.tensor([t.item() for t in target_result], device='cuda:0')
            slack_weights = torch.where(label == 1, slack_weight_1, slack_weight_0)
            #regularization = torch.mean(torch.where((result <= 0.2),
                                    #(result - 0.5) ** 2,
                                    #torch.zeros_like(result)))
            loss = (criterion(result, label) * slack_weights).mean() #+ alpha * regularization 
            val_loss += loss.item() * data.num_graphs
    
    avg_val_loss = val_loss / len(val_dataset)
    wandb.log({"Validation Loss": avg_val_loss, "epoch": epoch})
    print(f"Epoch {epoch + 1}, Validation Avg Loss: {avg_val_loss:.4f}")

    # Save checkpoint
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss,
        }, checkpoint_path)