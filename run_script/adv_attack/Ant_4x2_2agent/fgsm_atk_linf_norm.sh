#!/bin/bash


cd ../../../

ENV_ARGS="env_args.scenario=""Ant-v2"" env_args.agent_conf=""4x2"""


CONFIG="test_nepisode=50 \
		checkpoint_path=""results/marl_models/maddpg_ant"" \
		single_adv_model_path=""results/adv_models/adv_ddpg_ant""\
		seed=42 evaluate=True
		noise_constraint_type=""linf_norm""\
		adv_eps=[0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25]\
		fgsm_iter=30\
		"


# 0,1
NAME=fgsm_atk_linf_norm_ant_4x2_new_agent01
AGENT_CONFIG="attack_agent=[0,1] num_atk_agent=2"

python src/main.py --config=maddpg_adv_fgsm --env-config=mujoco_multi with ${ENV_ARGS} name=${NAME} ${CONFIG} ${AGENT_CONFIG}

# 0,2
NAME=fgsm_atk_linf_norm_ant_4x2_new_agent02
AGENT_CONFIG="attack_agent=[0,2] num_atk_agent=2"

python src/main.py --config=maddpg_adv_fgsm --env-config=mujoco_multi with ${ENV_ARGS} name=${NAME} ${CONFIG} ${AGENT_CONFIG}

# 0,3
NAME=fgsm_atk_linf_norm_ant_4x2_new_agent03
AGENT_CONFIG="attack_agent=[0,3] num_atk_agent=2"

python src/main.py --config=maddpg_adv_fgsm --env-config=mujoco_multi with ${ENV_ARGS} name=${NAME} ${CONFIG} ${AGENT_CONFIG}

# 1,2
NAME=fgsm_atk_linf_norm_ant_4x2_new_agent12
AGENT_CONFIG="attack_agent=[1,2] num_atk_agent=2"

python src/main.py --config=maddpg_adv_fgsm --env-config=mujoco_multi with ${ENV_ARGS} name=${NAME} ${CONFIG} ${AGENT_CONFIG}

# 1,3
NAME=fgsm_atk_linf_norm_ant_4x2_new_agent13
AGENT_CONFIG="attack_agent=[1,3] num_atk_agent=2"

python src/main.py --config=maddpg_adv_fgsm --env-config=mujoco_multi with ${ENV_ARGS} name=${NAME} ${CONFIG} ${AGENT_CONFIG}

# 2,3
NAME=fgsm_atk_linf_norm_ant_4x2_new_agent23
AGENT_CONFIG="attack_agent=[2,3] num_atk_agent=2"

python src/main.py --config=maddpg_adv_fgsm --env-config=mujoco_multi with ${ENV_ARGS} name=${NAME} ${CONFIG} ${AGENT_CONFIG}
