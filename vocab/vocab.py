import sentencepiece as spm

spm.SentencePieceTrainer.Train(
    '--input=../data/yo/train.yo ' 
    '--input_format=text '
    '--model_prefix=yoruba '
    '--vocab_size=30000 '
    '--train_extremely_large_corpus=true '
    '--shuffle_input_sentence=true '
    # '--character_coverage=0.9995 '
)