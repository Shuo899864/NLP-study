# 为什么提出chinesebert
因为汉语和英语有不一样的地方，汉字的字形就自带语义信息，并且汉字里有多音字，汉字的发音很大程度关联了语义。
# 模型
主要是在输入上，分为token embedding，字形embedding，拼音embedding，然后经过一个融合层，得到融合的embedding，再加上位置embedding，最后经过一个bert。
## 输入层
token embedding就是在汉字粒度上的词嵌入，这个和bert相同，尺寸是(batch, sq_len, emb_size)。<br>
字形embedding，使用三种字体的汉字图片，每张图片尺寸是24×24，将24×24×3的输入展开成长度2352的向量，然后经过一个线性层，尺寸同上。<br>
拼音embedding，拼音是一串字母加一个1-4的数字代表语调，使用一个卷积核宽度为2的CNN和max pooling，得到拼音embedding，尺寸同上。<br>
融合层，将以上三种embedding在最后一个维度concat起来，尺寸是(batch, sq_len, 3 × emb_size)，再经过一个线性层，得到尺寸是(batch, sq_len, emb_size)
# 训练
主要是一些普通的技巧，比如不仅MASK单字，还MASK词语，还有动态MASK，然后输入有时候是一个单句，有时候是一堆句子组合起来长度接近512，这些在别的论文里都见过了。
