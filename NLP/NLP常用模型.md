# 机器学习模型
## 决策树
## HMM/CRF
# 深度学习模型
## word2vec
word2vec是一种通过上下文训练词向量的方法，包含两个模型：CBOW（用上下文去预测中心词）和Skip-gram（用中心词预测上下文），和两种训练方式：负采样和层序softmax。<br>
负采样：选择中心词窗口内的词和不在中心词窗口内的词，分别与中心词组成样本进行训练。
层序softmax：输出层是霍夫曼树，霍夫曼树是一个二叉树，以语料中出现过的词当做叶子节点，以各词在语料中出现的次数当做权值进行构造，这样在每一层不断进行二分类，最后走到的叶子节点即为预测的单词。
```
class SkipGram(paddle.nn.Layers):
    def __init__(self, vocab_size, embedding_size, init_scale=0.1):
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.embedding = paddle.nn.Embedding(self.vocab_size, self.embedding_size)
        self.embedding_out = paddle.nn.Embedding(self.vocab_size, self.embedding_size)
                    
    def forward(self, context_words, target_words, label, eval_words):
        context_words_emb = self.embedding(context_words)
        target_words_emb = self.embedding_out(target_words)
        eval_words_emb = self.embedding(eval_words)
        
        word_sim = paddle.multiply(context_words_emb, target_words_emb)
        word_sim = paddle.sum(word_sim, dim = -1)
        pred = paddle.nn.sigmoid(word_sim)

        loss = paddle.nn.BCELoss(pred, label)
        loss = paddle.mean(loss)
        
        word_sim_on_fly = paddle.matmul(eval_words_emb, 
            self.embedding._w, transpose_y = True)

        return pred, loss, word_sim_on_fly
```
## Glove
## fasttext
## TextCNN
## RNN/LSTM
## Bert
