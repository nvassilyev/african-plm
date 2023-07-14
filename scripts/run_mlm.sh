# General version of run_mlm.sh

file="run_mlm.py"
output_dir="yo_mlm"
lang="yo"
model="../models/phase1_yo"
tokenizer="../models/phase1_yo"
src_lang="yo_XX"

mv ../data/mlm/${lang}/train.${lang} ../data/mlm/${lang}/train.txt
mv ../data/mlm/${lang}/eval.${lang} ../data/mlm/${lang}/eval.txt

CUDA_VISIBLE_DEVICES=0 python3 ../code/${file} \
    --model_name_or_path $model \
    --train_file ../data/mlm/${lang}/train.txt \
    --validation_file ../data/mlm/${lang}/eval.txt \
    --per_device_train_batch_size 8 \
    --do_train \
    --do_eval \
    --save_steps 125000 \
    --max_steps 125000 \
    --max_seq_length 256 \
    --overwrite_output_dir \
    --cache_dir ../.cache \
    --output_dir ../models/mlm/${output_dir} \
    --tokenizer_name $tokenizer \
    --src_lang $src_lang \
    # --num_train_epochs 3 \+