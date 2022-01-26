# 为什么提出bigbird
主要是基于transformer的模型计算和存储复杂度是序列长度的平方，bigbird提出一种稀疏注意力机制让复杂度降为线性复杂度。
# 模型结构
bigbird的注意力机制是以下三种注意力的组合。
## 随机注意力
每个token随机attend到r个token上。（理论依据大概就是随机图可以近似生成完全图）
## 窗口注意力
每个token attend到自身为中心w距离的token上（左右各w/2）。（理论依据就是局部相关性）
## 全局注意力
设计g个全局token，与句子里所有token可以互相attend。有两种做法，一种是现有token选择一部分作为全局token，一种是增加一部分token作为全局token。
# 复杂度分析
复杂度就是（r+w+g）× n，但是实验中rwg取值都比较大，很难说bigbird运算更快，但bigbird能处理更长的文本，还是有一定意义的。
