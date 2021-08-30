# Generate KITTI multiple-object dataset

EXP_NAME="results/gen_kitti_mulobj_train_5x5"
PY_NAME="${HOME_DIR}/dataloaders/kittimulobj_loader.py"
PHASE="gen_kittimulobj"
NUM_POINTS="2048"
KITTI_MULOBJ_NUM_ADD_MODEL="3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22"
KITTI_MULOBJ_NUM_EXAMPLE="3 19 79 264 716 1612 3044 4871 6642 7749 7749 6642 4871 3044 1612 716 264 79 19 3"
KITTI_MULOBJ_SCENE_RADIUS="5"
KITTI_MULOBJ_OUTPUT_FILE_NAME="${HOME_DIR}/dataset/kittimulobj/kitti_mulobj_param_train_5x5_2048"

RUN_ARGUMENTS="${PY_NAME} --exp_name ${EXP_NAME} --phase ${PHASE} --num_points ${NUM_POINTS} --kitti_mulobj_num_add_model ${KITTI_MULOBJ_NUM_ADD_MODEL} --kitti_mulobj_num_example ${KITTI_MULOBJ_NUM_EXAMPLE} --kitti_mulobj_scene_radius ${KITTI_MULOBJ_SCENE_RADIUS} --kitti_mulobj_output_file_name ${KITTI_MULOBJ_OUTPUT_FILE_NAME}"