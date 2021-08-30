# Counting experiment

EXP="count_tearing_kitti"
PY_NAME="${HOME_DIR}/experiments/counting.py"
EXP_NAME="results/${EXP}"
CHECKPOINT="${HOME_DIR}/results/train_tearing_kitti/"
PHASE="counting"
BATCH_SIZE="8"
COUNT_FOLD="4"
EPOCH_INTERVAL="-1 10"
CONFIG_FROM_CHECKPOINT="True"
DATASET_NAME="kitti_mulobj"
COUNT_SPLIT="test_5x5"
PRINT_FREQ="50"
GRID_DIMS="45 45"
SVM_PARAMS="100"

RUN_ARGUMENTS="${PY_NAME} --exp_name ${EXP_NAME} --checkpoint ${CHECKPOINT} --phase ${PHASE} --batch_size ${BATCH_SIZE} --count_fold ${COUNT_FOLD} --epoch_interval ${EPOCH_INTERVAL} --config_from_checkpoint ${CONFIG_FROM_CHECKPOINT} --dataset_name ${DATASET_NAME} --count_split ${COUNT_SPLIT} --print_freq ${PRINT_FREQ} --grid_dims ${GRID_DIMS} --svm_params ${SVM_PARAMS}"