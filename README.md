# Python-Protein-Protein-Interaction-GNN-Project
## Overview
This project uses the NetworkX and PyTorch libraries to display the relationships between proteins within the dataset and uses a graph convolutional network (GCN) built in PyTorch to evaluate whether two new proteins will interact with one another. Protein-protein interactions are key in predicting future interactions in drug development, the interconnected web of relationships and proteins relies on deep learning algorithms over regular machine learning algorithms to analyise and comprehend due to the hierarchical nature of the hidden layers within the GCN allowing it to find increasingly complex patterns as layers increase.

## Knowledge Graphs
A knowledge graph (KG) consists of two objects: nodes and edges. Nodes can represent almost anything, edges represent the relationship between two given nodes. Here, a node represents a protein and the existence of an edge implies that two proteins interact.

## Graph Convolutional Networks

A graph convolutional network (GCN) is a type of graph neural network (GNN), a GCN is made up of a few layers: a graph convolution, pooling and fully connected. The core of a GCN is in the name - the graph convolution layer, it aggregates data from a neighbouring nodes, for all nodes. The pooling layer decreases the dimensionality of the hidden, convoluted graph by discarding a percentage of the nodes which have the least information. The fully connected layer 


# Mathematics of the GCN
The mathematical equations within deep learning is not particularly complicated to write out, in practice however doing the actualy calculations is very complex due to the sheer size of the matricies.

The mathematics of a GCN is relatively simple to grasp: first, the graph is represented with two matricies, the **adjacency** and the **degree** matrix. The adjacency matrix (_square_) having values indicating the prescence of a node and the degree matrix (_diagonal_), with each diagonal value indicating how many edges a node has (also reffered to as the **degree** of a node).

Next is the actual main function of the GCN which is the GCNCONV (convolution) function:
$H^{(l+1)} = \sigma \left( D^{-\frac{1}{2}} A D^{-\frac{1}{2}} H^{(l)} W^{(l)} \right)$

Where $H^{l}$ is the matrix of node features, at layer $l$

$\sigma$ is the ReLU activation function (or some other non-linear activation function).

$D$ is the degree matrix

$W^{l}$ is the weight matrix, at layer $l$



# Code of the GCN

First, we must install and call the correct libraries for the project:


<img width="935" alt="Screenshot 2024-04-18 at 15 19 49" src="https://github.com/JIC1444/Python-GNN-Project/assets/158279190/cb027789-ba6e-4180-8afd-75b8e1007e0b">



Then, using the .nn.datasets library as well as the help of DataLoader, ClusterData and ClusterLoader, we extract the PPI (protein-protein interaction) dataset from the paper, getting the following values when printed: 


<img width="1182" alt="Screenshot 2024-04-18 at 15 20 14" src="https://github.com/JIC1444/Python-GNN-Project/assets/158279190/45c4fd67-2f09-4116-823a-aa9bff75678f">



We must then create the GCN, this consists of init and one core function: **forward**. 

The init function takes in the variables: **num_node_features**, **hidden_dimension** and **num_classes**. num_node_features gives the number of input features per node. Hidden_dimension gives the dimension of the representations of the data learned between the GCN layers eg. if your input num_node_features = 10 and you set hidden_dimension = 20 then this allows the ability for the GCN to recognise more complex patterns, however too many hidden dimensions can lead to overfitting. num_classes will be two, since the protein's interactions here are binary: they interact or they don't.

The forward function calls the two functions self.conv1,2 and uses the function ReLU. self.conv1,2 both take in their neighbours' information and combine it into a new node, however self.conv2 acts on a ReLU transformed matrix.


<img width="955" alt="Screenshot 2024-04-18 at 15 20 51" src="https://github.com/JIC1444/Python-GNN-Project/assets/158279190/fffb5aa6-6e1c-49b3-a12b-3077967cb266">



We then create an instance of the GCN, with num_node_features = 50, hidden_channels = 100, num_classes = 121


<img width="834" alt="Screenshot 2024-04-18 at 15 21 11" src="https://github.com/JIC1444/Python-GNN-Project/assets/158279190/1535a500-5f0e-4862-a18b-209ddcfb11a5">



We must then create a training and testing loop for the data using the GCN model created:


<img width="993" alt="Screenshot 2024-04-18 at 15 22 47" src="https://github.com/JIC1444/Python-GNN-Project/assets/158279190/420464a6-aa38-4977-803f-5be4562591c4">



And finally run the model, printing the results of each epoch:


<img width="1178" alt="Screenshot 2024-04-18 at 15 24 03" src="https://github.com/JIC1444/Python-GNN-Project/assets/158279190/eb524a30-0fbb-42d3-ac60-e7792bedf3b5">



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


From these results we can see that the loss is going down slightly, which is good. The Test values do go up from a below average score to just an average score, which is to be expected as this model is incredibly bearbones and not specialised or tuned in any way. I look to tuning the parameters of the model as the next step!
