# 为什么提出tine-bert
理由都是一样，bert太大了。
# 多维度蒸馏
总的损失函数是以下损失函数的加权平均。
## attention蒸馏
Lattn =1/hΣ(i=1 to h)MSE(ASi, ATi)：计算每个注意力头学生和教师之间的MSE loss，最后平均。
## hidden state蒸馏
Lhidn = MSE(HSWh, HT)：计算学生和教师隐状态之间的MSE loss，Wh是一个可学习的参数，用来把HS映射到与HT相同的尺寸。
## embedding蒸馏
Lembd = MSE(ESWe, ET)：计算学生和教师embedding之间的MSE loss，We是一个可学习的参数，用来把ES映射到ET相同的尺寸。
## output蒸馏
Lpred = CE(zT/t, zS/t)：计算学生和教师预测logits之间的CE loss，其中t代表softmax的温度参数。
# 两步蒸馏
## 常规蒸馏
就是那bert的预训练模型进行蒸馏，发现学生模型一般比教师模型差一些。
## 特定任务蒸馏
作者提出BERT的参数量对于特定任务来说是过多的，因此小模型完全可以在特定任务上取得更好的成绩。所以在特定任务上进行了数据增强。
## 数据增强
对于一句话中的每个词x：如果x是单个词，使用MASK替换x，并使用BERT预测MASK掉的词，取K个概率最大的词；如果x不是单个词，使用glove词向量找到语义最相近的K个词。最后按一定的替换概率从K个词中随机抽取一个替换。这里替换的概率是0.4，每句话最大替换数量是20，K是15.
