python run-v2.py \
    --model_name hfl/chinese-macbert-base \
    --ner_model_name uer/roberta-base-finetuned-cluener2020-chinese \
    --num_labels 2 \
    --data_dir data \
    --maxlength 128 \
    --pred_output_dir submissions-ner \
    --output_model_dir ner_run_2 \
    --epoch 2 \
    --batch_size 8 \
    --kfolds 10 \
    --lr 1e-5 \
    --alpha 0.3 \
    --gamma 0.8 \
    --perform_testing \
    --folds ner_run_2/folds.txt