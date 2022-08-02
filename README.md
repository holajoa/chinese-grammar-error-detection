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
    - 降低learning rate + 用 focal loss? 有改善, dev set F1 ~0.75 on 000 examples
      - ```python run.py --model_name hfl/chinese-macbert-base --num_labels 2 --data_dir .\data\  --maxlength 64 --adversarial_training_param 0 --pred_output_dir .\submissions\ --output_model_dir ./sample_run --epoch 3 --batch_size 8 --kfolds 4 --num_training_examples 1000 --perform_testing --lr 2e-5 --alpha 0.3 --gamma 0.5 ```
      - 调参：