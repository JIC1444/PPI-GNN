# Python-GNN-Project

This project at its core is the representation of protein-protein interactions from the dataset: (PyTorch) and uses the NetworkX library to display the relationships between proteins within the dataset and uses a Graph Neural Network (GNN) built in PyTorch. Protein-protein interactions are key in predicting future interactions in drug development, the interconnected web of relationships and proteins requires deep learning methods to analyise and comprehend due to the sheer size and complexity of the dataset, as well as the hidden insights neural networks bring forward.

A knowledge graph (KG) consists of two objects: nodes and edges. The nodes represent a thing, it can be almost anything imaginable, for example a movie or a country, here though it is a protein. An edge is the way something interacts with another, for example Pilot is an episode of a TV show, here "episode" is the interaction, however for my KG True or False will be the set of edges (True if the proteins interact, False if not). A node can have many edges but an edge may only link two nodes.

A **Graph Convolutional Netowork (GCN)** is a type of GNN, it takes in **node features** (basically a fingerprint for each individual node containing information about it) and the **edge indexes** (one edge index is one pair of nodes, all of the edge indexes map out the KG). 

Then, these inputs are ran through the layers of the GCN, which takes one at a time and compares it to its neighbours (called **aggregation**) and is done by either matrix summatation or multiplication to combine data from its neighbours. Multiple layers are used to allow deeper complexity, for example, layer 1 may only look at a node and its immediate surrounding nodes, however the next layer may look at all nodes within a one node gap and so on.

The output from the GCN is a node embeddingm which is the original node but transformed into an embedding vector, this represents all of the neighbour information about the node which was aggregated in the process before. Also a class score is given to each node, giving the likelihood of it belonging to a particular class eg.  

# Mathematics

The basics mathematics of a GCN is relatively simple to grasp: first, the graph is represented with two matricies, the **adjacency** and the **degree** matrix. The adjacency matrix (_square_) having values indicating the prescence of a node and the degree matrix (_diagonal_), with each diagonal value indicating how many edges a node has (also reffered to as the **degree** of a node).

Next is the actual main function of the GCN which is the GCNCONV (convolution) function: '''H^{(l+1)} = \sigma \left( D^{-\frac{1}{2}} A D^{-\frac{1}{2}} H^{(l)} W^{(l)} \right)'''


# Code

First, we must install and call the correct libraries for the project:


Then, using the .nn.datasets library as well as the help of DataLoader, ClusterData and ClusterLoader, we extract the PPI (protein-protein interaction) dataset from the paper, getting the following values when printed: 


We must then create the GCN, this consists of a two core functions: **__init__** and **forward**. 

The init function at the bare minimum takes in the variables: **num_node_features**, **hidden_dimension** and **num_classes**. num_node_features gives the number of input features per node. Hidden_dimension gives the dimension of the representations of the data learned between the GCN layers eg. if your input num_node_features = 10 and you set hidden_dimension = 20 then this allows the ability for the GCN to recognise more complex patterns, however too many hidden dimensions can lead to overfitting. num_classes will be two, since the protein's interactions here are binary: they interact or they don't. Extra variables can be added to customise the output of the GCN.

The forward function allows forward propagation of the model, it has the node feature matrix (A) and the edge indexes fed into it and transforms each of the values in the matrix with a weight matrix applied.


We must then run a training loop for the data using the GCN model created:


And finally we must evaluate our model using the class scores:

