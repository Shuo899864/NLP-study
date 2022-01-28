# 为什么提出ctrl
由于生成任务的不可控性，提出了一种基于控制码的条件语言模型。
# CTRL语言模型
传统语言模型是学习序列分布p(x)=∏p(xi|x<i)<br>
CTRL语言模型是学习在控制码作为生成前缀的序列分布p(x|c)=∏p(xi|x<i,c)
# 可控生成
## 采样
使用带温度的随即采样，并且每次从top k中取token，k是启发式的，让k个概率相加大于一个阈值，如果下一个token概率高，则k会变小。当有多个高概率候选时，采用贪婪地选择下一个token。对于可能产出重复的采样，要采用惩罚机制，如果token已经产生过，就对T施加一个系数θ（文中用的1.2）
## 控制码
按领域: Wiki，Books，Reviews，Horror，Relationships，Legal<br>
更复杂的控制码:Science Title, Politics Title, Running Text, Horror Text, Reviews Rating，不同的 Link 代表不同的特征（domain, subdomain, entities, entity relations, and even dates）<br>
触发特定任务的：问答、翻译<br>
Zero-shot code-mixing<br>
# 来源归因
使用pθ(c|x)∝pθ(x|c)p(c)计算，证明模型固有地依赖于原始的关联进行预测，不关心关联是否正确或好坏（事实表明，相互矛盾的主张经常出现在相同的上下文中）。CTRL 证明了特定的领域更可能包含与给定陈述相似的语言。
