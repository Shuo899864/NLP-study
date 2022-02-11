# 为什么提出ernie-gram
BERT的MLM任务每次MASK一个token，为了学到短语会MASK一个连续的n-gram，也就是每个token会用一个MASK符号代替，作者认为这种方式缺点在于每个token的预测是单独的，没有考虑token之间的关系，并且每个token是在整个词典进行预测选择，ernie-gram使用的方式是将n-gram用一个MASK符号替换，然后在一个n-gram词典上进行预测，为了充分学习语义，使用一种设计好的attention mask，在两种预测上同时训练。同时还采用了electra的思路，使用模型生成的token去替换原句子的token，让模型检测是否替换。
# 模型
## 连续MLM
输入序列是x，n-gram开头下标是b，z是n-gram序列，MLM从b中随机采样15%，然后把z中对应的n-gram mask掉，每个token替换成一个mask符号。
## 明确的n-gram mlm
输入n-gram实体序列是y，对于要mask掉的n-gram，每个n-gram替换成一个mask符号。
## 综合n-gram预测
使用一个mask符号，同时预测n-gram粒度和单个token粒度，损失函数是二者之和。<br>
具体实现，为了用单个mask符号预测一个n-gram里的所有token，使用[Mi]符号聚合上下文信息来预测第i个token。<br>
使用[Mi]作为Q，mask的输入序列作为K,V。比如在位置2有2个token要预测，就在输入序列后面跟[M1][M2]，[M1][M2]的position embedding跟位置2的相同。<br>
在进行n-gram粒度的预测时，要与token粒度的预测部分隔离，也就是在attention的softmax里面的scale的QK点积减去一个无穷大的值；在进行token粒度预测时，每个[Mi]可以attend到输入序列和它自己，不同的[Mi]之间不能互相attend。这样做的原因是长度信息对模型预测是有害的，会让模型武断的剔除一些语义与正确答案相近但长度不同的结果。
## 强化的n-gram关系建模
使用一个明确的n-gram mlm小模型产生n-gram预测，用预测的n-gram替换输入序列中原来的token，然后用标准的综合模型，分别从两个粒度预测原序列中的n-gram，损失函数是两个模型之和，让小模型尽可能生成正确token，让大模型尽可能恢复原序列。同时引入RTD替换检测任务训练大模型。
## n-gram的抽取
使用T检验，对于一个长度l，抽取所有的l-gram计算分数，并按分数排序，从大到小取k个加入到n-gram词表。

