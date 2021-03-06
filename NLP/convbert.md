# 动机
研究发现bert的自注意力机制存在计算冗余（bert的自注意力画出来集中在对角线附近），为了解决这一问题，convbert采用了动态卷积，动态卷积可以捕捉局部依赖；为了区分同一个词在不同上下文中的语义，又提出了基于跨度的动态卷积。为了降低计算量，把输入的embedding先投射到较低维度再输入自注意力；前馈神经网络使用了分组线性层。
# 模型
## 动态卷积
传统卷积是使用一个W（d×k，d是词嵌入维度，k是卷积核宽度）的卷积核。将卷积核沿着通道维度d绑定，得到轻量化卷积和LW（维度为k），与传统相比它减少了d的计算量。但卷积核的参数对于任何位置都是固定的，不好捕捉词语的多样性。<br>
动态卷积是使用当前token在线性层和门控线性单元（GLU）之后动态的生成一个卷积核，并与附近token进行卷积。动态卷积核DW=softmax(WfXi)。与自注意力相比效率更高，但卷积核的生成仅依赖单个token，对于不同上下文的同一个单词，卷积核是相同的，因此引入基于跨度的动态卷积。
## 基于跨度的动态卷积
使用深度可分离卷积来收集token跨度的信息，然后动态生成卷积内核。通过生成基于其局部上下文而不是单个token的输入token的局部关系，有助于内核有效地捕获局部依赖性。使用Q和V矩阵生成q和v，使用基于跨度的动态卷积生成k，q和k逐点相乘生成动态卷积核softmax(Wf(qk))，卷积核再与v进行卷积。
## 混合注意力块
把自注意力和卷积混合起来，其中QV是相同的，K是不同的，然后把两个注意力的结果concat起来
## 自注意力的瓶颈设计
有些头是多余的，因此先把输入投射到较低维度，减少头的数量。也就是说这里的QKV矩阵不再是正方形，如果投射出来维度变为一半，头的数目也会减少一半。
## 分组前馈
在词嵌入维度上分成多组进入线性层，最后的输出concat起来，减少计算量。
