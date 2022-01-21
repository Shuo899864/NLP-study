# 为什么提出Roberta
作者研究bert各种超参数对模型效果的影响，发现bert是没有训练充分的。于是做出以下改进：1. 训练更长时间，更大的batch size，更多的数据。2. 去掉NSP。3. 使用更长的样本训练。4. 训练的时候动态地MASK。
# 动态MASK
原始的BERT是数据处理的时候一次性MASK，所以每个epoch模型看到的数据是一样的，为了避免模型看到一样的数据，原始数据被复制了10次，然后MASK训练了40个epoch，相当于同样的数据模型看了4次。roberta使用的是动态MASK，句子在输入模型的时候才被MASK，实验证明动态MASK比静态MASK的效果略微好一点。
# 去掉NSP
segment-pair+NSP:原始BERT的训练方式，抽取两个片段进行训练。
sentence-pair+NSP：抽取两个句子进行训练，由于句子长度小于片段，因此提高了batch size，使总的token数接近segment。
full-sentences:直接抽取句子，使样本长度接近512，如果句子长度不够可以跨文档抽取。
doc-sentences：直接抽取句子，不跨文档抽取，因此如果句子长度不够512会动态的提高batch size。
结果是sentence-pair不如segment-pair，原因是破坏了长距离依赖。去掉NSP任务比保留NSP任务效果好，一个直观的解释，或者说猜测是因为，可能是Bert在消融实验去除NSP的时候，仍然保持的是原始的输入，即有NSP任务的时候的输入形式。doc-sentence比full-sentence效果好。
# 其他细节
更大的batch size（256对比8K），更多的数据（16G对比160G），训练更长时间（100M对比300/500K（batch size已经是32倍了））
