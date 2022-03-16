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
Betweenness centrality 概念為尋找哪一個點在一個 Graph 中屬於重要的樞紐位置為連接的重要橋梁，也就是透過點之間來計算最短距離。在實際的場域中Graph 會不斷地改變而且也有可能會不斷地變大，因此有來不及計算的問題，在這篇論文中主要針對 Betweenness centrality 作為探討，論文中提到比起實際算出 Betweenness centrality 正確的數值，得出那些點具有比較高的 BC才是我們所要關切的，因此就有了top-N% nodes的概念，再加上在每個時間點的 Graph 型態都不一樣，所以也引入了 Transfer learning ，目標在於希望可以透過學習多個 Graph 來找到在 Graph中具有 High BC value 節點的特性。此篇論文使用了 encoder-decoder ， encoder 用來將每個 nodes 映射到一個 embedding vector 而 decoder 則是將embedding vector 映射到 BC 的分數。 

### 2.	IMPLEMENTATION  
整體架構: 

![image](https://user-images.githubusercontent.com/51444652/158140318-bf941edf-d256-4992-aa62-009eff357ddc.png)

論文參數設置:

![image](https://user-images.githubusercontent.com/51444652/158140348-9fb80ca8-3f76-4e3e-9784-6a71f88e6c08.png)


#### 2.1	 Create Graph 

##### 2.1.1 Generating synthetic graph  
使用 nx.random_graphs.powerlaw_cluster_grap來生成圖片， 參數設置 with n=“number of nodes”, m=4, p=0.05 ，Graph follow by power-law distribution nx.random_graphs.powerlaw_cluster_graph(random.randint(500,800), 4, 0.05) 
透過生成的 Graph 取出 edge index 和 node 的資訊，在 edge index 中因為必須考慮到邊是屬於 bidirectional ，因此需要另外處理改成雙向的形式。在 node 資訊中，因為在 model 的 initial feature 長相為 degree[[n],1,1] ，所以需要對 node 做轉換。

##### 2.1.2 Calculate betweenness centrality
透過 nx.betweenness_centrality 計算點與點之間的 BC value ，因為產生出來的 output 會有太小的問題，會導致 model train 不起來，所以多加了 log 來收斂 。


#### 2.2	 Encoder
在 Encoder 使用三層的 GCN Layer ，embedding dimension 設置為 (128, 128)

##### 2.2.1 Neighborhood Aggregation - GCN
在Encoder 的部分，透過 Neighborhood aggregation models 的方式來得知每個點的 attributes ，好處在於節點之間的參數可以共享也可以在沒看過的節點中給予其 embedding vector 。

![image](https://user-images.githubusercontent.com/51444652/158143193-0b4084f0-f8c6-4583-93bc-1eff05ecde70.png)

在實作中使用  message passing 的方式 ，其概念跟 convolution filter很像 ， 透過相鄰點來求得點的特徵 。(如下) 

![image](https://user-images.githubusercontent.com/51444652/158143496-3b243491-b7ac-41ed-8833-9223fe63e401.png)

photo credit to :( https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial7/GNN_overview.html) 

Message passing 利用 Pytorch 中 CREATING MESSAGE PASSING NETWORKS 中所提供的 Implementing the GCN Layer 

##### 2.2.2 COMBINE Function 
為了獲取比較好的 feature ， combine 在Layer 中neighborhood 的 embedding 和上一層的 embedding，因此採用了 GRU gate的機制， update gate 用來選擇要記得前一層多少的資訊。(公式如下) 

![image](https://user-images.githubusercontent.com/51444652/158143841-3b596bee-d014-4a2e-bf1c-c4c76b7ed523.png)

作者比較了其他的 Combine function，發現使用 GRU 能夠取得較佳的特徵。

##### 2.2.3 Layer Aggregation 
論文中以element-wise的方式，取出最大值，得到一個128維的output。
計算 Betweennsess centrality 使用 nx.betweenness_centrality 來求 BC 的數值。

#### 2.3 Decoder
採用兩成的 hidden layer 和 LeakyReLU 將先前的 embedding 轉換為 score 。 

### 3 Training Algorithm

![image](https://user-images.githubusercontent.com/51444652/158565102-974b3365-9548-424d-91fb-0052b9068513.png)


#### 3.1 Pairwise ranking loss
Loss function

![image](https://user-images.githubusercontent.com/51444652/158565176-c974cae0-a67c-472c-8853-8f00e3c8dc26.png)

Sample node pair : 這裡採用論文的方式隨機 sample 5 個 node (一個點要與五個
點做比較) 。
使用 Kendall tau 來觀察其變數之間的相關程度 (ground truth value / predict value)，這裡使用 scipy.stats.kendalltau 來計算。
透過 model 運算過後得出 predict 的 BC value ，將預測的 BC value 與 ground  truth BC value 排序，取出 top 1 、top5 、top10 的 accuracy。

### 4.Experiment Result

![image](https://user-images.githubusercontent.com/51444652/158565319-fd056419-6d1d-4380-9a22-dd3d8705ed78.png)

#### 4.1 COMPARE 
利用 nx.random_graphs.powerlaw_cluster_graph(n, m, p) 來 Generate Graph 這裡將 n ( node 的個數)、 m (each node 的邊數) 、 p (Graph 形成三角形的機率) 做調整以觀察其中的變化 。

* 針對 node 個數調整 (800-1000)、(800-500)、(400-500)、(100-300) 
![image](https://user-images.githubusercontent.com/51444652/158565661-7f015462-76ae-4b32-9322-9c36f72d1326.png)


* 針對 number of random edges to add each new node : 2、4、8、10
![image](https://user-images.githubusercontent.com/51444652/158565734-2456d8de-6119-4749-a697-317fa4ae5450.png)


* p (Graph 形成三角形的機率) 做調整
![image](https://user-images.githubusercontent.com/51444652/158565884-7e825845-bde4-47b6-912d-e9b06c726f64.png)
![image](https://user-images.githubusercontent.com/51444652/158565937-def9e82e-1a84-4857-9700-23058ff30b0d.png)




