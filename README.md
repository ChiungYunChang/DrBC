# DrBC
Implement the DrBC approach from Learning to Identify High Betweenness Centrality Nodes from Scratch: A Novel Graph Neural Network Approach. paper : https://arxiv.org/pdf/1905.10418.pdf


* [1.INTRODUCTION](#1introduction)  
* [2.IMPLEMENTATION](#2implementation) 
  * [2.1 Create Graph](#21-create-graph)
    * [2.1.1 Generating synthetic graph](#211-generating-synthetic-graph)
    * [ 2.1.2 Calculate betweenness centrality](#212calculate-betweenness-centrality)
  * [2.2.Encoder](#22encoder)
    * [2.2.1 Neighborhood Aggregation - GCN](#221-neighborhood-aggregation---gcn)
    * [2.2.2 COMBINE Function](#222-combine-function) 
    * [2.2.3 Layer Aggregation](#223-layer-aggregation)
  * [2.3 Decoder](#23-decoder)
* [3. Training Algorithm](#3-training-algorithm)
  * [3.1 Pairwise ranking loss](#31-pairwise-ranking-loss)
* [4. Experiment Result](#4-experiment-result)
  * [4.1 COMPARE](#41-compare)


### 1.	INTRODUCTION  
The concept of Betweenness centrality is to identify which nodes in a graph serve as crucial bridges connecting different parts of the network, by calculating the shortest paths between nodes. In real-world scenarios, graphs are often dynamic and may continuously change or grow, posing challenges for timely calculations. This paper primarily focuses on exploring Betweenness centrality. Instead of computing the exact Betweenness centrality values, the paper emphasizes identifying nodes with higher BC values as the ones of interest. Hence, the concept of top-N% nodes is introduced. Additionally, since the graph structure varies at each time point, Transfer learning is introduced to learn characteristics of nodes with high BC values across multiple graphs. The paper employs an encoder-decoder architecture, where the encoder maps each node to an embedding vector, and the decoder maps the embedding vector to BC scores.

### 2.	IMPLEMENTATION  
Architecture: 

![image](https://user-images.githubusercontent.com/51444652/158140318-bf941edf-d256-4992-aa62-009eff357ddc.png)

Para:

![image](https://user-images.githubusercontent.com/51444652/158140348-9fb80ca8-3f76-4e3e-9784-6a71f88e6c08.png)


#### 2.1	 Create Graph 

##### 2.1.1 Generating synthetic graph  
To generate a graph using nx.random_graphs.powerlaw_cluster_graph, with parameters n as the number of nodes, m as 4, and p as 0.05,Graph follow by power-law distribution nx.random_graphs.powerlaw_cluster_graph(random.randint(500,800), 4, 0.05) 
To extract the edge index and node information from the generated graph, additional processing is required to handle bidirectional edges in the edge index. For the node information, transformation is necessary to fit the initial feature shape expected by the model, which is degree[[n],1,1].

##### 2.1.2 Calculate betweenness centrality
To calculate the Betweenness Centrality (BC) values between points using nx.betweenness_centrality, and mitigate issues with too small output values causing training problems by applying a logarithmic transformation for convergence

#### 2.2	 Encoder
To implement three layers of Graph Convolutional Network (GCN) in the encoder, with an embedding dimension set to (128, 128).

##### 2.2.1 Neighborhood Aggregation - GCN
In the Encoder section, utilize neighborhood aggregation models to determine the attributes of each node. The advantage of this approach lies in the ability to share parameters between nodes and provide embedding vectors for unseen nodes. By aggregating information from neighboring nodes, each node can obtain a representation that incorporates information from its local neighborhood. This allows for efficient parameter sharing and enables the model to generalize well to unseen nodes by leveraging information from their neighboring nodes during training.

![image](https://user-images.githubusercontent.com/51444652/158143193-0b4084f0-f8c6-4583-93bc-1eff05ecde70.png)

In the implementation, using message passing, which conceptually resembles convolutional filters, to compute node features by aggregating information from neighboring nodes. This method involves passing messages between adjacent nodes in the graph, allowing each node to update its feature representation based on the information received from its neighbors. Similar to convolutional filters aggregating information from neighboring pixels in an image, message passing enables nodes to incorporate information from their local neighborhood. By iteratively passing messages between nodes, each node can refine its feature representation by considering information from its surrounding nodes, resulting in richer and more informative node features. (shown in pic)

![image](https://user-images.githubusercontent.com/51444652/158143496-3b243491-b7ac-41ed-8833-9223fe63e401.png)

photo credit to :( https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial7/GNN_overview.html) 

Message passing utilizes the implementation of Graph Convolutional Networks (GCN) provided in PyTorch's "Creating Message Passing Networks" module.


##### 2.2.2 COMBINE Function 
To obtain better features, it combine the embeddings of the neighborhood and the embeddings from the previous layer in the layer. Therefore, we adopt the mechanism of GRU gates, where the update gate is used to decide how much information from the previous layer to retain. The formula is as follows:

![image](https://user-images.githubusercontent.com/51444652/158143841-3b596bee-d014-4a2e-bf1c-c4c76b7ed523.png)

(The author compared other combining functions and found that using the GRU mechanism resulted in obtaining better features.)
##### 2.2.3 Layer Aggregation 
In the paper, the maximum value is obtained using an element-wise approach to generate a 128-dimensional output. Betweenness centrality is calculated using nx.betweenness_centrality to determine the BC values.

#### 2.3 Decoder
A hidden layer with a 20% dropout rate and LeakyReLU activation function is employed to transform the previous embeddings into scores. 

### 3 Training Algorithm

![image](https://user-images.githubusercontent.com/51444652/158565102-974b3365-9548-424d-91fb-0052b9068513.png)


#### 3.1 Pairwise ranking loss
Loss function

![image](https://user-images.githubusercontent.com/51444652/158565176-c974cae0-a67c-472c-8853-8f00e3c8dc26.png)


Sample node pairs: Here, I randomly sample 5 node pairs using the method described in the paper (each node is compared with 5 other nodes). I use Kendall tau to observe the correlation between the ground truth values and predicted values. We calculate this using scipy.stats.kendalltau. After performing computations using the model, I obtain predicted BC values, then sort the predicted BC values and ground truth BC values and calculate the accuracy for the top 1, top 5, and top 10 predictions.

### 4.Experiment Result

![image](https://user-images.githubusercontent.com/51444652/158565319-fd056419-6d1d-4380-9a22-dd3d8705ed78.png)




