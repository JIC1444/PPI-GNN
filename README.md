# Python-GNN-Project

This project at its core is the representation of protein-protein interactions from the dataset: (PyTorch) and uses the NetworkX library to display the relationships between proteins within the dataset and uses a Graph Neural Network (GNN) built in PyTorch. Protein-protein interactions are key in predicting future interactions in drug development, the interconnected web of relationships and proteins requires deep learning methods to analyise and comprehend due to the sheer size and complexity of the dataset.

A knowledge graph (KG) consists of two objects: nodes and edges. The nodes represent a thing, it can be almost anything imaginable, for example a movie or a country, here though it is a protein. An edge is the way something interacts with another, for example Pilot is an episode of a TV show, here "episode" is the interaction, however for my KG True or False will be the set of edges (True if the proteins interact, False if not). A node can have many edges but an edge may only link two nodes.

A **Graph Convolutional Netowork (GCN)** is a type of GNN, it takes in **node features** (basically a fingerprint for each individual node containing information about it) and the **edge indexes** (one edge index is one pair of nodes, all of the edge indexes map out the KG). 
Then, these inputs are ran through the layers of the GCN, which takes one at a time and compares it to its neighbours (called **aggregation**) and is done by either matrix summatation or multiplication to combine data from its neighbours. Multiple layers are used to allow deeper complexity, for example, layer 1 may only look at a node and its immediate surrounding nodes, however the next layer may look at all nodes within a one node gap and so on.
The output from the GCN is a node embeddingm which is the original node but transformed into an embedding vector, this represents all of the neighbour information about the node which was aggregated in the process before. Also a class score is given to each node, giving the likelihood of it belonging to a particular class eg.  


# **Code**

First, we must install and call the correct libraries for the project:


Then, using the .nn.datasets library, we extract the PPI (protein-protein interaction) dataset from the paper: 


We must then preprocess the data, making sure it is in node, edge form:


We must then create the GCN, this consists of a two core functions: __init__ and forward. The init function at the bare minimum takes in the variables: num_node_features, hidden_dimension and num_classes. Where num_node_features gives
and the forward function allows forward propagation of the model, which essentially means


We must then run a training loop for the data using the GCN model created:


And finally we must evaluate our model using the class scores:

