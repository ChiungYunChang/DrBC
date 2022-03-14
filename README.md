# DrBC
Implement the DrBC approach from Learning to Identify High Betweenness Centrality Nodes from Scratch: A Novel Graph Neural Network Approach. paper : https://arxiv.org/pdf/1905.10418.pdf

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


## training result 
![image](https://user-images.githubusercontent.com/51444652/158065393-a22e9e26-da53-458f-af6c-3efad2bee752.png)
