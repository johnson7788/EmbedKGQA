# Relation matching module

UPDATE: Please see https://github.com/malllabiisc/EmbedKGQA/issues/69 for an issue related to a missing file.

如论文第4.4.1节所述，一个模型被训练来检测与问题相关的关系，这个模型被进一步用于关系匹配。

## Model

文件`pruning_main.py`训练这个模型。在可下载的pretrained_models.zip中，有一个预训练的版本。


```python
pruning_dataloader.py
pruning_main.py  精简的训练文件, 就是训练问题等价于哪些关系
pruning_model.py
pruning_test.txt   #测试集
pruning_train.txt  # 训练集
```

## Relation matching scoring

请看ipython笔记本`relation_matching_eval.ipynb`，用于对关系匹配的QA模型输出进行评分。这个笔记本假设基础QA模型的分数存储在
基准QA模型的分数被存储在`webqsp_scores_full_kg.pkl`文件中。这可以通过在`main.py`的validate函数中设置`writeCandidatesToFile = True`来实现。
然后用这些分数以及上述模型来计算最终的答案分数。
