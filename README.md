# Python-Protein-Protein-Interaction-GNN-Project
## Overview
This project is aimed for those with limited experience in PyTorch and no exposure to Graph Neural Networks (GNNs). Using mainly the NetworkX and PyTorch libraries to display the relationships between proteins within a dataset with a graph, then train a neural network to predict whether two new proteins will interact with one another. 

Protein-protein interactions are key in drug development, the interconnected web of relationships and proteins relies on deep learning algorithms over classical machine learning algorithms to analyise and comprehend due to the hierarchical nature of the hidden layers within the GNN, allowing it to find increasingly complex patterns with deeper and deeper layers.

## Knowledge Graphs
A knowledge graph (KG) consists of two objects: nodes and edges. Nodes can represent almost anything, edges represent the relationship between two given nodes. In this PPI dataset, a **node** represents a **protein** and the existence of an **edge** implies that **two proteins interact**.

Mathematically a graph is written $G = (V, E, \boldsymbol{A})$ where $V$ is the set of nodes, $E$ is the edges between each pair of nodes and $\mathbf{A}$ is the adjacency matrix. This is useful notation when observing the graph convolution function.

## Graph Convolutional Networks
A Graph Convolutional Network (GCN) is a type of GNN. A GCN is made up of a few layers: a graph convolution layer (commonly either spectral, spatial or attention), a pooling layer and a fully connected layer (also reffered to as a multi-layer perceptron or linear layer). 

The core of a GCN is in the graph convolution layer, it aggregates data from neighbouring nodes to produce a very complex matrix representation of each nodes and the neighbouring information. 

The pooling layer then decreases the dimensionality of the hidden graph by discarding a percentage of the nodes which have the least information, this decreases computational strain by discarding weak or nonsensical connections in the data. 

The fully connected (FC) layer has every neuron in the FC layer, connected to every neuron in the input layer. In each neuron then a value is computed, consisting of a weighted sum of the inputs and the addition of a bias vector, where finally this value is input into an activation function to introduce non-linearity. This 'block' of functions can then be repeated n-times depending on the writer's choice, and often there are many FC layers at the end of these n convolutional blocks.

The output is usually an m-dimensional vector where each i'th element in the output vector can be assigned to any class value, probability or _one-hot_ value. A one-hot encoded vector is a vector with m-1 0s and a singular 1, where the 1 represents the label of the data. It is explained easier with an example, in our case,  there are 121 unique proteins in the dataset, this is layed out in a particular order. Since m = 121, there will be a 1 in the place that the protein is found in this specified order, hence 120 0s in all other places.

## Mathematics of the Model Layers
The mathematical equations within deep learning is not particularly complicated to write out and are mostly vector or matrix-based computations. In practice however doing the actualy calculations is extremely complex due to the sheer size of the matricies.

There are several inputs to the GCN: 

The mathematics of a GCN is relatively simple to grasp: first, the graph is represented with two matricies, the **adjacency** and the **degree** matrix. The adjacency matrix (_square_) having values indicating the prescence of a node and the degree matrix (_diagonal_), with each diagonal value indicating how many edges a node has (also reffered to as the **degree** of a node).

Next is the actual main function of the GCN which is the GraphConv function:\
\
$H^{(l+1)} = \sigma \left( D^{-\frac{1}{2}} A D^{-\frac{1}{2}} H^{(l)} W^{(l)} \right)$

> Where $H^{(l)}$ is the matrix of node features, at layer $l$
> 
> $\sigma$ is the ReLU activation function (or some other non-linear activation function).
> 
> $D$ is the degree matrix.
> 
> $A$ is the
> 
> $W^{l}$ is the weight matrix, at layer $l$.
> 

The core operation in the equation is **feature propagation**, where node information is aggregated from neighboring nodes. This consists of:

_Normalisation_: The term $D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$ normalises the adjacency matrix. Without this, nodes with higher degrees could disproportionately impact the feature aggregation - this normalisation ensures consistent scaling of features.

_Aggregation_: Multiplying $A$ with $H^{(l)}$ propagates information from neighboring nodes to each node. This step captures the graph structure and relationships between nodes.

_Transformation_: $H^{(l)}W^{(l)}$ applies a linear transformation to the aggregated features, allowing the model to learn  representations of the data.

_Activation_: Finally, applying $\sigma$ introduces non-linearity, allowing the network to learn complex patterns beyond linear relationships.

This process is layer-wise, meaning at the input layer, $H^{(0)}$ is the raw feature matrix of the graph nodes. Then as the model moves through the layers, the features $H^{(l)}$ are updated using the equation above, incorporating data from nodes further from each other in the graph. The final layer produces the output $H^{(L)}$ , which can be used for predicting new data points, such as node classification, graph classification, or edge/relationship predictions.


# Code and Method Walkthrough

### Install and call the correct libraries for the project:


```{p}
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
```

### Using the  library, extract the PPI dataset from the paper "":

```{p}
# Import and split data from PyTorch.
dataset = PPI(root=".../GCN DATA")  
data = dataset[0]

path = os.join(os.dirname(os.realpath(__file__)), '..', 'data', 'PPI')
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')

train_data = Batch.from_data_list(train_dataset) 
# DataBatch(x=[44906,50], edge_index=[2,1226368], y=[44906,121], batch=[44906], ptr=[21])

cluster_data = ClusterData(train_data, num_parts=4,
                           recursive=False,
                           save_dir=train_dataset.processed_dir) 
# ClusterData(50)

# Dataloaders load the dataset as torch_geometric objects.
train_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False) 
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# Set device depending on availibility.
device = "cuda" if torch.cuda.is_available() else "cpu"    
```

### Create the GCN architecture, this consists of init and one core function: **forward**. 

The init function takes in the variables: **num_node_features**, **hidden_dimension** and **num_classes**. num_node_features gives the number of input features per node. Hidden_dimension gives the dimension of the representations of the data learned between the GCN layers eg. if your input num_node_features = 10 and you set hidden_dimension = 20 then this allows the ability for the GCN to recognise more complex patterns, however too many hidden dimensions can lead to overfitting. num_classes will be two, since the protein's interactions here are binary: they interact or they don't.

The forward function calls the two functions self.conv1,2 and uses the non-linear activation function ReLU (Rectified Linear Units). self.conv1,2 both take in their neighbours' information and combine it into a new node, however self.conv2 acts on a ReLU transformed matrix.

```{p}
# Simple starting GCN to test the complexity of the task at hand.
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

# Define the loss function, used to evaluate model performance.
def loss_function(output, target):
     return F.binary_cross_entropy_with_logits(output, target)

# Initialize the model with chosen parameters as a base.
model = GCN(50, 100, 121).to(device)
optimiser = optim.Adam(model.parameters(), lr=0.01)
```

_The parameter hidden_channels can be modified later if this model is observed to over or underfit to the data._


### Training and testing functions for the model:

```{p}
def train():
    # Set the model to training mode, this ensures the wieghts are updated through the optimisation of a loss function.
    model.train()
    total_loss = 0
    for d in train_loader:
        data = d.to(device) 
        optimiser.zero_grad() # Clear gradients.
        # Use the model to predict the labels.
        output = model(data.x, data.edge_index)
        # Load the actual label of the data.
        label = data.y[:output.size(0)]
        loss = loss_function(output, target)

        loss.backward() # Compute gradients for this iteration.
        optimiser.step() # Update the model weights.

        total_loss += loss.item() * data.num_nodes
    return total_loss/train_data.num_nodes

@torch.no_grad() # Decorator used so no gradients are computed.
def test(loader):
    model.eval() # Model in evaluation mode makes sure model wieghts remain constant.
    labels, preds = [], []
    for d in loader:
        labels.append(data.y)
        output = model(data.x.to(device), data.edge_index.to(device))
        preds.append((output > 0).float().cpu())
    label, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    f1 = f1_score(y, pred, average='micro') if pred.sum() > 0 else 0 
    return f1 # Return only non-negative values of the F1 score.
```



### Run the model, printing results of each epoch:

```{p}
if __name__ == "__main__":
    for epoch in range(1, 200):
        loss = train() 
        val_f1 = test(val_loader) 
        test_f1 = test(test_loader)
        print(f"Epoch: {epoch}, Loss: {loss:.4f}, Val: {val_f1:.4f}, Test: {test_f1:.4f}")
```


**Results**

Epoch: 1, Loss: 0.1736, Val: 0.4058, Test: 0.4090
Epoch: 2, Loss: 0.1665, Val: 0.4073, Test: 0.4091
Epoch: 3, Loss: 0.1629, Val: 0.4108, Test: 0.4112
Epoch: 4, Loss: 0.1593, Val: 0.4153, Test: 0.4159
.      .        .            .             .
.      .        .            .             .
.      .        .            .             .
Epoch: 196, Loss: 0.1202, Val: 0.5493, Test: 0.5556
Epoch: 197, Loss: 0.1182, Val: 0.5428, Test: 0.5481
Epoch: 198, Loss: 0.1199, Val: 0.5417, Test: 0.5474
Epoch: 199, Loss: 0.1198, Val: 0.5461, Test: 0.5523

## Observations
The loss actually starts out at a relatively low value and decreases by about 33%. This is a good start!

The validation and the test accuracies increase by 0.15, this is also a good start but not quite satisfactory, ending up at an accuracy of just 55%. 

To determine if the model is **overfitting or underfitting** we can use the values of the training and validation loss. If the validation loss is _high_ but the training loss is _low_, this is a sign of overfitting (since it shows that the model does well on seen data, but fails to generalise and predict new data). If both loss values are similarly high and static, this indicates underfitting (the model does not effectivley capture any of the relationships in the data).

If we apply these rules, then it seems that the model could be underfitting, due to the low F1 accuracy score in the test and the limited movement 

If you made it this far, then congrats - you have just created your first Graph Neural Network architecture!

## Extensions
**Extension 1**: How could the parameters be modified to improve the performance?

**Extension 2**: How could the model be modified to no longer underfit to the data?
_Hint: think about how a model can recognise more complex patterns when increasing layers_









