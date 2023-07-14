# General version of run_ner.sh

# output_dir="fairseq_ner"
# lang="yo"
# model="../models/xmod_fairseq"
# tokenizer="../tokenizer"
# src_lang="yo_XX"

output_dir=$1
lang=$2
model=$3
tokenizer=$4
src_lang=$5

# max_length=164
# /mnt/disk/masakhane-ner/data

for j in 1 2 3 4 5
do  
    echo "Training ${output_dir} with seed $j"
	export MAX_LENGTH=200
	export BERT_MODEL=$model
	export OUTPUT_DIR=$output_dir
	export TEXT_RESULT=test_result$j.txt
	export TEXT_PREDICTION=test_predictions$j.txt
	export BATCH_SIZE=32
	export NUM_EPOCHS=20
	export SAVE_STEPS=10000
	export SEED=$j
	CUDA_VISIBLE_DEVICES=0,1 python3 ../code/train_ner.py --data_dir ../data/ner/yo \
	--model_type xmod \
	--model_name_or_path ${BERT_MODEL} \
	--output_dir ../models/${OUTPUT_DIR} \
	--test_result_file $TEXT_RESULT \
	--test_prediction_file $TEXT_PREDICTION \
	--max_seq_length  $MAX_LENGTH \
	--num_train_epochs $NUM_EPOCHS \
	--per_gpu_train_batch_size $BATCH_SIZE \
	--save_steps $SAVE_STEPS \
	--seed $SEED \
	--do_train \
	--do_eval \
	--do_predict \
	--cache_dir ../.cache \
	--overwrite_output_dir \
	--src_lang ${src_lang} \
	--freeze \
    --tokenizer_name $tokenizer 
done

find "$model" -type f ! -regex "^.*\.txt$" -delete

config_file="$output_dir/config.txt"
echo "lang=$lang" > "$config_file"
echo "base_model=$model" >> "$config_file"
echo "tokenizer=$tokenizer" >> "$config_file"
echo "src_lang=$src_lang" >> "$config_file"
