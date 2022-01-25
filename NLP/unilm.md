# 为什么提出unilm
因为BERT是双向语言模型，在预测单词的时候能看到它左右的上下文，不适合进行文本生成任务。unilm设计了多种任务使同一个模型能同时具备NLG和NLU的能力。
# unilm的三种任务
单向LM（包括从左到右和从右到左）/双向LM/seq2seq LM<br>
## 单向LM
使用attention mask，形状如同一个上三角矩阵，从左到右的话，右上角的值被设置成-∞，也就是预测单词的时候只能看到它自己和左边的单词。
## 双向LM
attention mask矩阵都是0，预测单词能看到它左右的所有单词。
## seq2seq LM
输入是两个句子，用sep符号隔开，句子一的attention mask是双向的（都为0），句子二的attention mask是单向的，也就是句子一预测的时候可以看到左右的单词，句子二预测的时候能看到句子一和句子二左边的单词。
# 其他细节
mask的时候15% 的有被替换的概率，其中80%被真正替换。在这80%真正替换的里面有80%单个token被替换，20%的二元或者三元tokens被替换