## 为什么提出Transformer：
1. 目前主流的序列生成模型是基于RNN或CNN，RNN的机制让他无法很好的并行化，序列长度过长影响比较大，并且内存占用也限制了批量的大小。<br>
2. 作者这里提出了Transformer，一种只基于注意力的模型，注意力机制允许对序列中的依赖关系建模，而不考虑距离关系。实验证明这个模型有很好的效果，很高的并行化度，并且训练时间更短。减少顺序计算也为GPU运算奠定了基础，在Transformer中，任意两个位置通过O(1)的操作复杂度连接。<br>
## Transformer的结构
### encoder和decoder
主流的序列生成模型都包含编码器和解码器，编码器把符号形式的输入序列映射到连续空间表达的序列，解码器生成一个序列，每次生成一个符号，这属于一种自回归模型，每次输出前会结合以前的输出和输入信息。<br>
Transformer也是使用这个结构来设计的，编码器包含6个一样的编码器层，每个层包含两个子层，分别是多头自注意力层和前馈神经网络层，两层都使用了残差连接和normalization，为了对齐维度，Transformer的所有层，包括嵌入层，都保持了相同的输出维度dmodel=512。解码器包含6个一样的解码器层，每个层包含三个子层，分别是带遮挡的多头自注意力层，多头交互注意力层，前馈神经网络层。带遮挡的多头自注意力层可以确保在解码的时候模型只能看到之前生成的序列，解码器的多头交互注意力层使用解码器的query和编码器最后一层输出的key和value。<br>
### 注意力机制
注意力机制是把query，key，value映射到输出，输出是利用q,k计算的权重对v进行加权和。Transformer使用的是缩放的点积注意力，首先，q,k具有相同的维度dk，使用q对所有的k计算点积，并除以根号dk进行缩放，缩放的原因是，如果dk比较大，那么点积也会比较大，会让softmax函数处于饱和区，梯度会比较小。对缩放后的点积施加softmax，获得v的权重。<br>
多头注意力：多头注意力意图提取不同特征子空间的信息，对于h个头，每个头的qkv是由输入乘一个dmodel*dk(dv)的权重矩阵生成，最后再拼接起来乘一个hdv*dmodel的权重矩阵。比如h=8个头，dk=dv=dmodel/8=64<br>
Transformer注意力的三种应用方式：编码器的qkv来自同一个内容，即原始输入或上一层编码器的输出，可以看到所有位置。解码器的自注意力看不到后面的位置，通过把相应位置的softmax输出设为负无穷。交互注意力层的q来自上一层的解码器，kv来自最后一层编码器的输出。<br>
### 前馈神经网络层
前馈神经网络层：两层全连接，对seq_len维度。第一层维度从512变成2048，第二层从2048变回512.<br>
### Layer Norm
Layer norm：也是对seq_len维度做normalization，原因一个是如果从batch维度看样本不等长，第二个是如果用每个位置的token作为标准化的特征，语义上差别是比较大的，不稳定。<br>
### 位置编码
位置编码：由于模型没有顺序结构，因此在输入中显式的加入位置编码，PE(pos,2i) = sin(pos/100002i/dmodel)，PE(pos,2i+1) = cos(pos/100002i/dmodel)，对于pos位置的第2i个嵌入维度，使用第一个公式，第2i+1个嵌入维度，使用第二个公式。作者在这里假设模型可以通过位置编码学习到相对位置（事实上在线性变换后，相对位置信息丢失了）<br>
