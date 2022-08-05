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
    --perform_testing \
    --lr 2e-5 \
    --alpha 0.4 \
    --gamma 1
```
比较稳定。

### 想法
1. 用POS tagging模型out-of-the-box检测句子结构问题 - 成分残缺
2. 开箱用困惑度模型，检测搭配？
3. 