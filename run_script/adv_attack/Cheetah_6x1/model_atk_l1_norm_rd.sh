#!/bin/bash


cd ../../../

ENV_ARGS="env_args.scenario=""HalfCheetah-v2"" env_args.agent_conf=""6x1"""


CONFIG="pgd_lr=0.5 pgd_step=30 plan_step=1 test_nepisode=50 \
		dynamic_model_path="""results/dynamic_models/HalfCheetah_6x1/model.pt""" \
		checkpoint_path=""results/marl_models/maddpg_cheetah_6x1"" \
		init_type=epsilon seed=42 evaluate=True pgd_solver=adam\
		noise_constraint_type=""l1_norm""\
		adv_eps=[0,0.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.,5.5,6.,6.5,7.,7.5,8.,8.5,9.,9.5,10.,10.]\
		"


NAME=maddpg_model_atk_l1_norm_cheetah_6x1_agent_rd
AGENT_CONFIG="attack_agent=[0,1,2,3,4,5]"

python src/main.py --config=maddpg_adv_model --env-config=mujoco_multi with ${ENV_ARGS} name=${NAME} ${CONFIG} ${AGENT_CONFIG}
