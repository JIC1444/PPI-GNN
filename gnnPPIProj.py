import os.path as os
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import ClusterData, ClusterLoader, DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import PPI
import torch.optim as optim
from sklearn.metrics import f1_score

#Import and split Data from PyTorch
dataset = PPI(root="VSCode Projects/GCN DATA")  
data = dataset[0]

path = os.join(os.dirname(os.realpath(__file__)), '..', 'data', 'PPI')
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')

train_data = Batch.from_data_list(train_dataset) 
#DataBatch(x=[44906,50], edge_index=[2,1226368], y=[44906,121], batch=[44906], ptr=[21])

cluster_data = ClusterData(train_data, num_parts=4,
                           , recursive=False,
                           save_dir=train_dataset.processed_dir) 
#ClusterData(50)
train_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=True,
                             num_workers=0)
#Geom. object ClusterLoader

val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False) #Geom. object DataLoader
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False) #Geom. object DataLoader

if torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")

#Bare bones GCN creation
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x) 
        x = self.conv2(x, edge_index) 
        return x

def calculate_loss(output, target):
     return F.binary_cross_entropy_with_logits(output, target)

model = GCN(50, 100, 121).to(device)
optimiser = optim.Adam(model.parameters(), lr=0.01)

#Train model
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device) 
        optimiser.zero_grad() #Clear previous gradients
        output = model(data.x, data.edge_index) #Forward through the model
        target = data.y[:output.size(0)]
        loss = calculate_loss(output, target)
        loss.backward() #Compute gradients
        optimiser.step() #Update model params
        total_loss += loss.item() * data.num_nodes
        return total_loss/train_data.num_nodes

@torch.no_grad() #Decorator used so no gradients are computed
#Test model
def test(loader):
    model.eval()
    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        out = model(data.x.to(device), data.edge_index.to(device))
        preds.append((out > 0).float().cpu())
    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    for epoch in range(1, 200):
        loss = train() #Train function performed
        val_f1 = test(val_loader) #Test function performed on actual data
        test_f1 = test(test_loader) #Test function performed with test data
        print(f"Epoch: {epoch}, Loss: {loss:.4f}, Val: {val_f1:.4f}, Test: {test_f1:.4f}")