# 为什么提出ernie
简单来讲，就是BERT在MLM任务中随机的MASK单词，但是没有融入先验知识，比如Harry Potter这个两个字的人名，MASK掉其中一个就应该通过另一个推测出来，还有Harry Potter和J. K. Rowling，应该通过二者的关系互相推测出来。ERNIE使用了实体级别的MASK策略和短语级别的MASK策略，让模型可以学到先验知识。
# 如何在模型中融入知识
ernie使用了三种不同级别的MASK：基本的词元级别MASK，英文每次MASK一个单词，中文每次MASK一个汉字；短语级别MASK，每次MASK一个短语，短语是使用词法分析和分词等工具提取出来的；实体级别MASK，每次MASK一个实体，实体包括人名地名组织名等。
# ernie在对话任务的做法
主要是把输入的token type id改成了dialogue embedding（Q和R），并且支持多轮对话。同样是预测被MASK的单词。
# 实验
主要结论是使用多种MASK策略比BERT词元级MASK效果好，使用多种来源的语料效果比单一预料效果好。最后这个完形填空测试说明ERNIE是可以学到实体级别知识的。
