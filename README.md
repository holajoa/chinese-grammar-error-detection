# chinese-grammar-error-detection
讯飞中文语义病句识别挑战赛。https://challenge.xfyun.cn/topic/info?type=sick-sentence-discrimination

Predicting all samples as positive:

```
python run.py --model_name hfl/chinese-macbert-base --num_labels 2 --data_dir .\data\  --adversarial_training_param 2 --pred_output_dir .\submissions\ --output_model_dir ./sample_run --epoch 4 --batch_size 8 --kfolds 4 --split_words --num_training_examples 1000
```

```
python run.py --model_name hfl/chinese-macbert-base --num_labels 2 --data_dir .\data\  --adversarial_training_param 2 --pred_output_dir .\submissions\ --output_model_dir ./sample_run --epoch 4 --batch_size 8 --kfolds 4  --num_training_examples 1000 --maxlength 64 --split_words 
```

**问题：only predict 1s**

**解决：**
1. 看下训练loss - 好像会上升。
    - 病句：正确句子=7:3，病句太多，只predict语病也能在dev set上取得不错结果。
    - 降低learning rate + 用 focal loss? 有改善, dev set F1 on 1000 examples ~0.8
      - 观察：f1分数浮动很大
      - 调参：
        - 一定不能加对抗训练！！加了就只会报1 - 正确的句子加入随机扰动，很有可能就变成带语病的了。
        - Focal loss的`gamma` 取0.5就好，再大也只会报1了
        - 去标点？有recall极低的情况。为什么？
        - 分词？f1略低于不分，推测是因为分词有的时候会复制词，对语病检测有伤害（比如用词重复）
    - **错标sample观察：**
      - 正确句子被标为病句更多。
      - 成分残缺误判几乎没有。
      - 很多新闻陈述类型的正确句子被误标为病句。
        - *国家发改委有关负责人在答记者问时指出，今年春运期间铁路票价不上浮，公路票价可以在政府指导价规定幅度内适当浮动。*
        - *叙利亚人权观察组织说，导弹不仅击中了空军基地内的机动防空系统，还命中了基地附近的一座民兵军营，炸死了十四个人。*
        - *中国体育健儿正在积极备战2016年奥运会，他们将在赛场上努力拼搏，争创佳绩。*
        - 解决：NER模型识别专有名词、年份等，含有此类句子的应该更难判断为病句。
      - 多重否定是难点
```
python run.py \
    --model_name hfl/chinese-macbert-base \
    --num_labels 2 \
    --data_dir ./data \
    --maxlength 64 \
    --pred_output_dir ./submissions \
    --output_model_dir ./sample_run \
    --epoch 3 \
    --batch_size 16 \
    --kfolds 4 \
    --num_training_examples 1000 \
    --lr 1e-5 \
    --alpha 0.3 \
    --gamma 0.5 \
    --perform_testing 
```
比较稳定。


### 改进
#### 新闻类句子被错误识别为病句
*Approach 1:* 开箱用NER模型，专有名词出现多的句子，
- focal loss进行调整（NER score高 -> focal loss gamma取接近0） focal_loss中的gamma改为gamma*1-(score**5)(1-mean_NER_score)
  - 不可以，因为prediction阶段不需要计算loss
- 用每个词的confidence score在hidden states上进行调整，然后再放进MLP head训练？
  - 尝试把ner model的hidden states输出和分类模型的相加（看看加个weight），然后再训练MLP head



### 未完成想法
1. 用POS tagging模型out-of-the-box检测句子结构问题 - 成分残缺
2. 开箱用困惑度模型，检测搭配？
3. 