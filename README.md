# DrBC
Implement the DrBC approach from Learning to Identify High Betweenness Centrality Nodes from Scratch: A Novel Graph Neural Network Approach. paper : https://arxiv.org/pdf/1905.10418.pdf

### 1.	INTRODUCTION  
Betweenness centrality 概念為尋找哪一個點在一個 Graph 中屬於重要的樞紐位置為連接的重要橋梁，也就是透過點之間來計算最短距離。在實際的場域中Graph 會不斷地改變而且也有可能會不斷地變大，因此有來不及計算的問題，在這篇論文中主要針對 Betweenness centrality 作為探討，論文中提到比起實際算出 Betweenness centrality 正確的數值，得出那些點具有比較高的 BC才是我們所要關切的，因此就有了top-N% nodes的概念，再加上在每個時間點的 Graph 型態都不一樣，所以也引入了 Transfer learning ，目標在於希望可以透過學習多個 Graph 來找到在 Graph中具有 High BC value 節點的特性。此篇論文使用了 encoder-decoder ， encoder 用來將每個 nodes 映射到一個 embedding vector 而 decoder 則是將embedding vector 映射到 BC 的分數。 

### 2.	IMPLEMENTATION  
整體架構: 

![image](https://user-images.githubusercontent.com/51444652/158140318-bf941edf-d256-4992-aa62-009eff357ddc.png)

論文參數設置: 
![image](https://user-images.githubusercontent.com/51444652/158140348-9fb80ca8-3f76-4e3e-9784-6a71f88e6c08.png)

## training result 
![image](https://user-images.githubusercontent.com/51444652/158065393-a22e9e26-da53-458f-af6c-3efad2bee752.png)
