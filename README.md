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
    - 降低learning rate + 用 focal loss? 有改善, dev set F1 on 1000 examples ~0.8 ***（但是F1score给正负样本权重相同，和focal loss不一致。怎么办？）***
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
*Approach 1:* 开箱用NER模型，理解专有名词
- focal loss进行调整（NER score高 -> focal loss gamma取接近0） focal_loss中的gamma改为gamma*1-(score**5)(1-mean_NER_score)
  - 不可以，因为prediction阶段不需要计算loss
- 用每个词的confidence score在hidden states上进行调整，然后再放进MLP head训练？
  - 尝试把ner model的hidden states输出和分类模型的concatenate，然后再训练MLP head 
    - 这样focal loss 参数要调整
    - lr=1e-5
    - best model metric 用 F1
- 不用Easy ensemble - F1 capped at ~80
  
初步实验结果：
- FP/FN $\approx 2:1$
    改进想法：gamma和alpha调参。目前：
      ```
      python run-v2.py \
        --model_name hfl/chinese-macbert-base \
        --ner_model_name uer/roberta-base-finetuned-cluener2020-chinese \
        --num_labels 2 \
        --data_dir data \
        --maxlength 128 \
        --pred_output_dir submissions-ner \
        --output_model_dir ner_run \
        --epoch 3 \
        --batch_size 8 \
        --kfolds 10 \
        --lr 1e-5 \
        --alpha 0.4 \
        --gamma 0.8 \
        --perform_testing \
        --best_by_f1 \
        --num_training_examples 5000 \
        --add_up_hiddens \
      ```
- 数据增强
- 每个词输出一个二分类，然后取左右logits差最大的一项做最终summarised logits输出。dev set上有提升

### 未完成想法
1. 数据增强
   1. 随机替换同义词：
    '雨珠砸在玻璃上，发出噼噼啪啪的声响。', 
    '雨珠砸在玻璃上，发出噼噼啪啪的声音。', 
    前者错误，后者正确
2. Classification head改成CRF？NONONONO
3. 缩句？POS困惑度 
   - https://huggingface.co/ckiplab/bert-base-chinese-pos
   - https://huggingface.co/KoichiYasuoka/chinese-bert-wwm-ext-upos
4. CLS与后面error_char一致的为简单样本，不一致的为困难样本，困难样本是否可以进行二次训练
5. 用正确的句子训练pos tagger + clf，然后用clf层预测likelihood