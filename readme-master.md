#EmbedKGQA
代码的论文名字：ACL 2020 paper Improving Multi-hop Question Answering over Knowledge Graphs using Knowledge Base Embeddings (Slides)
更新：新增了关系匹配的代码，请阅读readme的细节，如何使用他们

# 介绍：
## 数据和预训练模型
为了运行代码，首先下载data.zip和pretrained_model.zip，并把它们解压到主目录下
更新：关于WebQSP test的设置有些小问题，已经被修复，修复脚本：file qa_test_webqsp_fixed.txt 放在目录：data/QA_data/WebQuestionsSP
### MetaQA数据集
改变到目录./KGQA/LSTM。下面是一个例子如何运行这个QA训练代码
```buildoutcfg
python3 main.py --mode train --relation_dim 200 --hidden_dim 256 \
--gpu 2 --freeze 0 --batch_size 128 --validate_every 5 --hops 2 --lr 0.0005 --entdrop 0.1 --reldrop 0.2  --scoredrop 0.2 \
--decay 1.0 --model ComplEx --patience 5 --ls 0.0 --kg_type half
```
### WebQuestionsSP数据集
改变到目录./KGQA/RoBERTa。下面是一个例子如何运行这个QA训练代码
```buildoutcfg
python3 main.py --mode train --relation_dim 200 --do_batch_norm 1 \
--gpu 2 --freeze 1 --batch_size 16 --validate_every 10 --hops webqsp_half --lr 0.00002 --entdrop 0.0 --reldrop 0.0 --scoredrop 0.0 \
--decay 1.0 --model ComplEx --patience 20 --ls 0.05 --l3_reg 0.001 --nb_epochs 200 --outfile half_fbwq
```
注意：这将在没有关系匹配的vanilla设置中运行代码，关系匹配将不得不单独进行。关于关系匹配的细节可以在这里找到。表3中的数字是关系匹配后的数字。
另外，请注意，这个实现使用了通过libkge（https://github.com/uma-pi1/kge）创建的嵌入。这是一个非常有用的库，我建议你通过它来训练嵌入，
因为它支持稀疏嵌入+共享负采样，以加快像Freebase这样的大型KG的学习速度。

# 数据集生成
## MetaQA数据集
### KG图谱数据
有2个数据集：MetaQA_full 和 MetaQA_half，MetaQA_full数据集包含原始的kb.txt作为train.txt，并删除了重复的三元组。
 MetaQA_half数据集只包含50%的三元组（随机抽取，不替换）。
在半数数据集的train.txt中，有一些像 "entity NOOP entity "的行。这是因为在删除这些三元组时，该实体的所有三元组都被删除了，
因此任何KG嵌入实现都不会在train.txt文件中找到它们的任何嵌入矢量。通过包括这样的 "NOOP "三元组，我们并没有从KG中包括任何关于它们的额外信息，
它的存在只是为了让我们可以直接使用任何嵌入实现来为它们生成一些随机的向量。
### QA问答数据
每个数据集有5个文件 (1, 2 and 3 跳)
qa_train_{n}hop_train.txt
qa_train_{n}hop_train_half.txt
qa_train_{n}hop_train_old.txt
qa_dev_{n}hop.txt
qa_test_{n}hop.txt
其中，qa_dev、qa_test和qa_train_{n}hop_old分别与MetaQA的原始dev、test和train文件完全相同
对于qa_train_{n}hop_train和qa_train_{n}hop_train_half，我们以（头部实体，问题，答案）的形式添加了三重（h, r, t）。
这是为了防止模型在使用QA数据集训练QA模型时 "忘记 "实体嵌入。qa_train.txt包含所有的三元组，而qa_train_half.txt只包含MetaQA_half的三元组。
## WebQuestionsSP数据集
### KG图谱数据
有2个数据集: fbwq_full and fbwq_half
生成fbwq_full数据集: 我们将知识库限制为Freebase的一个子集，其中包含WebQuestionsSP的问题中提到的任何实体的2跳以内的所有事实。
我们进一步修剪，只包含那些在数据集中提到的关系。这个较小的KB有180万个实体和570万个三元组。
生成fbwq_half数据集: 我们从fbwq_full中随机抽取50%的三元组

### QA问答数据
与原始的WebQuestionsSP QA数据集相同。

