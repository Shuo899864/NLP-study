# 为什么提出ernie-doc
简单来讲，基于transformer的模型能处理的最大文本长度一般是512，因为计算的复杂度是文本长度的平方。对于长文本，如果简单的截取片段，或者使用稀疏注意力，会丢失片段间的上下文信息。所以提出了ernie-doc，包含强化的复现机制和回溯机制，可以处理长文本。ernie-doc使用了一个segment recording的训练目标，用来训练模型捕捉句子间的关系。
# 模型实现
## 稀疏transformer和重现transformer的做法
对于每层的state，稀疏transformer不做额外处理，重现transformer使用上一段文本的state和当前文本的state concat起来，这个concat后的state用来计算下一层的key和value，但是query只使用当前文本的state来计算，但是这种concat并不能让前面的文本获取后面的上下文信息。
## 回溯馈入机制
就是模拟人类阅读，一段文本读两次，第一次是简略的扫描，把每段文本的state缓存起来，记作H（seq_len × segments × layers × dmodel），第二次是回溯，利用缓存的state构造新的state，H先于上一段文本的state concat，再与当前文本的state concat。也就是说计算过程跟上面一样，再计算下一层的key/value之前，除了上一段文本的state再多concat一个所有文本合并的上下文信息H。这样做会造成较大的内存消耗，因此做了简化，只使用第N层的state，并且间隔N个segment取一个state，得到简化后的H（L × T/N × dmodel）
## 强化的复现机制
为了更好的利用前面的机制，作者想让h直接融入，因此对公式做了改动，也就是说recurrence使用上一个segment的同层state融合去计算下一层，这里改成了用上一个segment的下一层state融合去计算下一层，也就是使用过去的高层信息去丰富未来的低层信息。
## segment recording训练
除了MLM，增加了新的训练目标，也就是打乱一个文章的段落，让模型预测正确顺序，用来学习段落间的关系。对于K个段落，有K!个label，将各个段落依次输入，模型输出一个label
# 实验
主要关注消融实验的结果，没有segment recording，降低1个多点，没有回溯馈入，降低不到1个点，没有强化复现，降低2个多点，没有复现，降低3个多点。
