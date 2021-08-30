# Generate CAD model multiple-object dataset

EXP_NAME="results/gen_cad_mulobj_train_5x5"
PY_NAME="${HOME_DIR}/dataloaders/cadmulobj_loader.py"
PHASE="gen_cadmultiobj"
NUM_POINTS="2048"
AUGMENTATION="True"
CAD_MULOBJ_NUM_ADD_MODEL="3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22"
CAD_MULOBJ_NUM_EXAMPLE="3 19 79 264 716 1612 3044 4871 6642 7749 7749 6642 4871 3044 1612 716 264 79 19 3"
CAD_MULOBJ_NUM_AVA_MODEL="1133"
CAD_MULOBJ_SCENE_RADIUS="5"
CAD_MULOBJ_OUTPUT_FILE_NAME="${HOME_DIR}/dataset/cadmulobj/cad_mulobj_param_train_5x5"

RUN_ARGUMENTS="${PY_NAME} --exp_name ${EXP_NAME} --phase ${PHASE} --num_points ${NUM_POINTS} --augmentation ${AUGMENTATION} --cad_mulobj_num_add_model ${CAD_MULOBJ_NUM_ADD_MODEL} --cad_mulobj_num_example ${CAD_MULOBJ_NUM_EXAMPLE} --cad_mulobj_num_ava_model ${CAD_MULOBJ_NUM_AVA_MODEL} --cad_mulobj_scene_radius ${CAD_MULOBJ_SCENE_RADIUS} --cad_mulobj_output_file_name ${CAD_MULOBJ_OUTPUT_FILE_NAME}"