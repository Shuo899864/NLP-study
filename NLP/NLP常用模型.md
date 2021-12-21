# 机器学习模型
## 决策树
## HMM/CRF
# 深度学习模型
## 词向量模型
### word2vec
word2vec是一种通过上下文训练词向量的方法，包含两个模型：CBOW（用上下文去预测中心词）和Skip-gram（用中心词预测上下文），和两种训练方式：负采样和层序softmax。<br>
原始方法：输入层（N*vocab维度的one-hot），隐藏层（vocab*emb_size），输出层（emb_size*vocab），使用softmax增加正确单词的输出值，降低错误单词的输出值。<br>
负采样：由于词表过大导致softmax计算缓慢，且每次需要更新整个参数矩阵，因此把原问题优化为二分类问题，可以每次只更新一部分参数。选择中心词窗口内的词和不在中心词窗口内的词，分别与中心词组成样本进行训练。输入层（N*vocab维度的one-hot），隐藏层（两个矩阵，分为中心词的词向量和背景词的词向量，均为vocab*emb_size），输出层（两个词向量点乘+sigmoid）<br>
层序softmax：输出层是霍夫曼树，霍夫曼树是一个二叉树，以语料中出现过的词当做叶子节点，以各词在语料中出现的次数当做权值进行构造，这样在每一层不断进行二分类，最后走到的叶子节点即为预测的单词。每次只更新路径上的参数，达到节省计算量的目的。<br>
训练模型（使用负采样）：
```
import paddle


class SkipGram(paddle.nn.Layer):
    def __init__(self, vocab_size, embedding_size, init_scale=0.1):
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.embedding = paddle.nn.Embedding(
            self.vocab_size, self.embedding_size)

        self.embedding_out = paddle.nn.Embedding(
            self.vocab_size, self.embedding_size)

    def forward(self, center_words, target_words, label):
        center_words_emb = self.embedding(center_words)
        target_words_emb = self.embedding_out(target_words)

        word_sim = paddle.multiply(center_words_emb, target_words_emb)
        word_sim = paddle.sum(word_sim, axis=1)

        pred = paddle.nn.functional.sigmoid(word_sim)

        loss = paddle.nn.functional.binary_cross_entropy(word_sim, label)

        return pred, loss

```
### Glove
根据词频学习词共现矩阵X，X中的元素x_ij代表词j在词i的环境中出现的次数。
### fasttext
## TextCNN
## RNN/LSTM
## Bert
