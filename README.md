# Python-Protein-Protein-Interaction-GNN-Project
# Overview
This project at its core is the representation of protein-protein interactions from the dataset: (PyTorch) and uses the NetworkX library to display the relationships between proteins within the dataset and uses a Graph Neural Network (GNN) built in PyTorch. Protein-protein interactions are key in predicting future interactions in drug development, the interconnected web of relationships and proteins requires deep learning methods to analyise and comprehend due to the sheer size and complexity of the dataset, as well as the hidden insights neural networks bring forward.

A knowledge graph (KG) consists of two objects: nodes and edges. The nodes represent a thing, it can be almost anything imaginable, for example a movie or a country, here though it is a protein. An edge is the way something interacts with another, for example Pilot is an episode of a TV show, here "episode" is the interaction, however for my KG True or False will be the set of edges (True if the proteins interact, False if not). A node can have many edges but an edge may only link two nodes.

A **Graph Convolutional Netowork (GCN)** is a type of GNN, it takes in **node features** (basically a fingerprint for each individual node containing information about it) and the **edge indexes** (one edge index is one pair of nodes, all of the edge indexes map out the KG). 

Then, these inputs are ran through the layers of the GCN, which takes one at a time and compares it to its neighbours (called **aggregation**) and is done by either matrix summatation or multiplication to combine data from its neighbours. Multiple layers are used to allow deeper complexity, for example, layer 1 may only look at a node and its immediate surrounding nodes, however the next layer may look at all nodes within a one node gap and so on.

The output from the GCN is a node embeddingm which is the original node but transformed into an embedding vector, this represents all of the neighbour information about the node which was aggregated in the process before. Also a class score is given to each node, giving the likelihood of it belonging to a particular class eg.  

# Mathematics
The basics mathematics of a GCN is relatively simple to grasp: first, the graph is represented with two matricies, the **adjacency** and the **degree** matrix. The adjacency matrix (_square_) having values indicating the prescence of a node and the degree matrix (_diagonal_), with each diagonal value indicating how many edges a node has (also reffered to as the **degree** of a node).

Next is the actual main function of the GCN which is the GCNCONV (convolution) function: ```latex
H^{(l+1)} = \sigma \left( D^{-\frac{1}{2}} A D^{-\frac{1}{2}} H^{(l)} W^{(l)} \right)```

Where ```latexH^{l}``` is the matrix of node features, at layer l

```latex\sigma``` is the Relu function (non-linear activation function)

D is the degree matrix

```latexW^{l}``` is the weight matrix, at layer l

# Code

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


From these results we can see that the loss is going down slightly, which is good. The Test values do go up from a below average score to just an average score, which is to be expected as this model is incredibly bearbones and not specialised or tuned in any way.
