python run-v2.py \
    --model_name hfl/chinese-macbert-base \
    --ner_model_name uer/roberta-base-finetuned-cluener2020-chinese \
    --num_labels 2 \
    --data_dir ./data \
    --maxlength 64 \
    --pred_output_dir ./submissions-ner \
    --output_model_dir ./ner_run_ee \
    --epoch 4 \
    --batch_size 16 \
    --kfolds 8 \
    --lr 2e-5 \
    --easy_ensemble \
    --alpha 0.3 \
    --gamma 2 \
    --perform_testing \

