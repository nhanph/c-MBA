#!/bin/bash


cd ../../../

ENV_ARGS="env_args.scenario=""Ant-v2"" env_args.agent_conf=""4x2"""


CONFIG="test_nepisode=50 \
		checkpoint_path=""results/marl_models/maddpg_ant"" \
		single_adv_model_path=""results/adv_models/adv_ddpg_ant""\
		seed=42 evaluate=True
		noise_constraint_type=""l1_norm""\
		adv_eps=[0,0.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.,5.5,6.,6.5,7.,7.5,8.,8.5,9.,9.5,10.,10.]\
		fgsm_iter=30\
		"
		

for agent in 0 1 2 3
  do
	NAME=fgsm_atk_l1_norm_ant_4x2_new_agent${agent}
	AGENT_CONFIG="attack_agent=${agent}"

	python src/main.py --config=maddpg_adv_fgsm --env-config=mujoco_multi with ${ENV_ARGS} name=${NAME} ${CONFIG} ${AGENT_CONFIG}

done
