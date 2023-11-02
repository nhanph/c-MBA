import datetime
from functools import partial
from math import ceil
import numpy as np
import os
import pprint
import time
import threading
import torch as th
from gym import spaces
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

from envs import REGISTRY as env_REGISTRY

import pickle

# from utils.obs_utils import get_obs_info

def run(_run, _config, _log, pymongo_client):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    print('Use device {}'.format(args.device))

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    if pymongo_client is not None:
        print("Attempting to close mongodb client")
        pymongo_client.close()
        print("Mongodb client closed")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner, logger):

    # if args.log_episode:
        # unique_token = "{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    for eps in args.adv_eps:        
        time_elapse = 0
        time_list = []
        for episode in range(args.test_nepisode):
            start = time.time()
            ep_batch = runner.eval(epsilon=float(eps), test_mode=True)
            end = time.time()
            time_list.append(end-start)
            time_elapse += time_list[-1]
            # runner.close_env()
            if args.log_episode:
                if args.env_args['env_monitor']:
                    save_path = os.path.join(args.env_args['video_path'], 'log_episodes')
                else:
                    save_path = os.path.join(args.local_results_path, 'log_episodes', args.env_args['scenario'],args.name,"epsilon_"+str(float(eps)))
                os.makedirs(save_path, exist_ok=True)
                pickle.dump(ep_batch, open(os.path.join(save_path,"episode_"+str(episode)), "wb"))
                print('Episode saved to {}'.format(os.path.join(save_path,"episode_"+str(episode))))

        # print('Time elapsed: {:.2f} seconds, Time per episode: {:.2f} ({:.2f})'.format(time_elapse, np.mean(time_list), np.std(time_list)))
        msg = 'Time elapsed: {:.2f} seconds, Time per episode: {:.2f} ({:.2f})'.format(time_elapse, np.mean(time_list), np.std(time_list))
        logger.print_msg(msg)
    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def evaluate_dynamic_model(args, runner, logger):

    print('Evaluate dynamic model')
    for _ in range(args.test_nepisode):
        runner.dynamic_model_eval(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    #args.action_space = env_info["action_space"]
    args.action_spaces = env_info["action_spaces"]
    args.actions_dtype = env_info["actions_dtype"]
    args.normalise_actions = env_info.get("normalise_actions",
                                                False) # if true, action vectors need to sum to one


    # create function scaling agent action tensors to and from range [0,1]
    ttype = th.FloatTensor if not args.use_cuda else th.cuda.FloatTensor
    mult_coef_tensor = ttype(args.n_agents, args.n_actions)
    action_min_tensor = ttype(args.n_agents, args.n_actions)
    if all([isinstance(act_space, spaces.Box) for act_space in args.action_spaces]):
        for _aid in range(args.n_agents):
            for _actid in range(args.action_spaces[_aid].shape[0]):
                _action_min = args.action_spaces[_aid].low[_actid]
                _action_max = args.action_spaces[_aid].high[_actid]
                mult_coef_tensor[_aid, _actid] = np.asscalar(_action_max - _action_min)
                action_min_tensor[_aid, _actid] = np.asscalar(_action_min)
    elif all([isinstance(act_space, spaces.Tuple) for act_space in args.action_spaces]):    # NOTE: This was added to handle scenarios like simple_reference since the action space is Tuple
        for _aid in range(args.n_agents):
            for _actid in range(args.action_spaces[_aid].spaces[0].shape[0]):
                _action_min = args.action_spaces[_aid].spaces[0].low[_actid]
                _action_max = args.action_spaces[_aid].spaces[0].high[_actid]
                mult_coef_tensor[_aid, _actid] = np.asscalar(_action_max - _action_min)
                action_min_tensor[_aid, _actid] = np.asscalar(_action_min)
            for _actid in range(args.action_spaces[_aid].spaces[1].shape[0]):
                _action_min = args.action_spaces[_aid].spaces[1].low[_actid]
                _action_max = args.action_spaces[_aid].spaces[1].high[_actid]
                tmp_idx = _actid + args.action_spaces[_aid].spaces[0].shape[0]
                mult_coef_tensor[_aid, tmp_idx] = np.asscalar(_action_max - _action_min)
                action_min_tensor[_aid, tmp_idx] = np.asscalar(_action_min)

    args.actions2unit_coef = mult_coef_tensor
    args.actions2unit_coef_cpu = mult_coef_tensor.cpu()
    args.actions2unit_coef_numpy = mult_coef_tensor.cpu().numpy()
    args.actions_min = action_min_tensor
    args.actions_min_cpu = action_min_tensor.cpu()
    args.actions_min_numpy = action_min_tensor.cpu().numpy()

    def actions_to_unit_box(actions):
        if isinstance(actions, np.ndarray):
            return args.actions2unit_coef_numpy * actions + args.actions_min_numpy
        elif actions.is_cuda:
            return args.actions2unit_coef * actions + args.actions_min
        else:
            return args.args.actions2unit_coef_cpu  * actions + args.actions_min_cpu

    def actions_from_unit_box(actions):
        if isinstance(actions, np.ndarray):
            return th.div((actions - args.actions_min_numpy), args.actions2unit_coef_numpy)
        elif actions.is_cuda:
            return th.div((actions - args.actions_min), args.actions2unit_coef)
        else:
            return th.div((actions - args.actions_min_cpu), args.actions2unit_coef_cpu)

    # make conversion functions globally available
    args.actions2unit = actions_to_unit_box
    args.unit2actions = actions_from_unit_box

    action_dtype = th.long if not args.actions_dtype == np.float32 else th.float
    if all([isinstance(act_space, spaces.Box) for act_space in args.action_spaces]):
        actions_vshape = 1 if not args.actions_dtype == np.float32 else max([i.shape[0] for i in args.action_spaces])
    elif all([isinstance(act_space, spaces.Tuple) for act_space in args.action_spaces]):
        actions_vshape = 1 if not args.actions_dtype == np.float32 else \
                                       max([i.spaces[0].shape[0] + i.spaces[1].shape[0] for i in args.action_spaces])
    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "noise": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (actions_vshape,), "group": "agents", "dtype": action_dtype},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }

    if not args.actions_dtype == np.float32:
        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
        }
    else:
        preprocess = {}

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1 if args.runner_scope == "episodic" else 2,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if getattr(args, 'collect_transition', False):
            collect_transition = args.collect_transition
        else:
            collect_transition = False

        if getattr(args, 'eval_dynamic_model', False):
            if args.eval_dynamic_model:
                evaluate_dynamic_model(args, runner, logger)
                return

        if collect_transition == False and (args.evaluate or args.save_replay):
            evaluate_sequential(args, runner, logger)
            return


    # start collecting
    logger.console_logger.info("Beginning collecting for {:1.0e} timesteps".format(args.t_max))

    num_episodes = 0
    res_dict = {}

    total_ts = 0
    MAX_TIME_STEP = args.t_max

    last_count = 0

    data_folder = args.collect_save_path

    import pickle
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    save_path = os.path.join(data_folder, args.env_args['scenario'], unique_token)
    os.makedirs(save_path, exist_ok=True)
    n_chunk = 0

    while total_ts < MAX_TIME_STEP:    
        res_dict[num_episodes], ts = runner.collect()
        total_ts += ts
        num_episodes += 1

        if total_ts - last_count >= 1e4:
            last_count =total_ts
            logger.console_logger.info('Progess: {}/{}, {} chunks'.format(total_ts,MAX_TIME_STEP, n_chunk))
            logger.console_logger.info("Saving current batch to {}".format(save_path))
            pickle.dump(res_dict, open(os.path.join(save_path,"data_"+str(n_chunk)), "wb"))
            n_chunk += 1

            del res_dict
            res_dict = {}
            num_episodes = 0

    logger.console_logger.info("Done collecting data, total episodes: {}".format(num_episodes))
    runner.close_env()
    logger.console_logger.info("Finished Training")

def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
