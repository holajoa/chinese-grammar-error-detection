# chinese-grammar-error-detection
讯飞中文语义病句识别挑战赛。https://challenge.xfyun.cn/topic/info?type=sick-sentence-discrimination

```
python run.py --model_name hfl/chinese-macbert-base --num_labels 2 --data_dir .\data\  --adversarial_training_param 2 --pred_output_dir .\submissions\ --output_model_dir ./sample_run --epoch 4 --batch_size 8 --kfolds 4 --split_words --num_training_examples 1000
```