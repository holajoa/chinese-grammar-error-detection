python run.py \
    --model_name hfl/chinese-macbert-base \
    --num_labels 2 \
    --data_dir ./data \
    --maxlength 64 \
    --pred_output_dir ./submissions \
    --output_model_dir ./sample_run \
    --epoch 4 \
    --batch_size 16 \
    --kfolds 8 \
    --lr 1e-5 \
    --alpha 0.3 \
    --gamma 0.5 \
    --perform_testing 
    # --num_training_examples 1000 \