

## g_max_predictions_per_seq approx_to g_max_seq_length * g_masked_lm_prob

# online or offline
export train_mode=offline
export param_name=param1
export g_train_batch_size=128
export g_num_train_steps=10000
export g_max_seq_length=128
export g_max_predictions_per_seq=20
export g_masked_lm_prob=0.15
export g_dupe_factor=3

sh -x scripts/run_train_bert.sh  > log/$param_name.log &

# online or offline
export train_mode=offline
export param_name=param2
export g_train_batch_size=64
export g_num_train_steps=10000
export g_max_seq_length=128
export g_max_predictions_per_seq=20
export g_masked_lm_prob=0.15
export g_dupe_factor=3

sh -x scripts/run_train_bert.sh  > log/$param_name.log &

# online or offline
export train_mode=offline
export param_name=param3
export g_train_batch_size=128
export g_num_train_steps=10000
export g_max_seq_length=128
export g_max_predictions_per_seq=8
export g_masked_lm_prob=0.05
export g_dupe_factor=5

sh -x scripts/run_train_bert.sh  > log/$param_name.log &

# online or offline
export train_mode=offline
export param_name=param4
export g_train_batch_size=64
export g_num_train_steps=10000
export g_max_seq_length=128
export g_max_predictions_per_seq=8
export g_masked_lm_prob=0.05
export g_dupe_factor=5

sh -x scripts/run_train_bert.sh  > log/$param_name.log &

# online or offline
export train_mode=offline
export param_name=param5
export g_train_batch_size=32
export g_num_train_steps=10000
export g_max_seq_length=128
export g_max_predictions_per_seq=20
export g_masked_lm_prob=0.15
export g_dupe_factor=3

sh -x scripts/run_train_bert.sh  > log/$param_name.log &

# online or offline
export train_mode=offline
export param_name=param6
export g_train_batch_size=32
export g_num_train_steps=10000
export g_max_seq_length=128
export g_max_predictions_per_seq=8
export g_masked_lm_prob=0.05
export g_dupe_factor=5

sh -x scripts/run_train_bert.sh  > log/$param_name.log &

wait
