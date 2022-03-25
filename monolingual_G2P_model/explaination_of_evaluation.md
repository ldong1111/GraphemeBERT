## Explaination of evaluation
The comparison between the  tokenized gold phoneme sequence and the tokenized predicted phoneme sequence is not equivalent to the original gold phoneme sequence and predicted phoneme sequence. The different will only show when the tokenized gold phoneme sequence show some "\<unk\>" tokens (i.e., tokenized number is 0 ). To avoid this, we replace the \<unk\> index (0) to -1 make all "\<unk\>" of the gold phoneme sequence unpredictable, for both WER and PER comparison. 

