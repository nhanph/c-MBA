from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch
import copy
import time

class EvalRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](env_args=self.args.env_args, args=args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

        self.state_est_errors = []

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(   EpisodeBatch, 
                                    scheme, 
                                    groups, 
                                    self.batch_size, 
                                    self.episode_limit + 1,
                                    preprocess=preprocess, 
                                    device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            # self.env.render()
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            actions, info = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            if self.args.env in ["particle"]:
                cpu_actions = copy.deepcopy(actions).to("cpu").numpy()
                reward, terminated, env_info = self.env.step(cpu_actions[0])
                if isinstance(reward, (list, tuple)):
                    assert (reward[1:] == reward[:-1]), "reward has to be cooperative!"
                    reward = reward[0]
                episode_return += reward
            else:
                reward, terminated, env_info = self.env.step(actions[0].cpu())
                episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if self.args.action_selector is not None and hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def eval(self, test_mode=True, epsilon=0.0):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        noise_info = {
            'l1_norm': 0.,
            'l2_norm': 0.,
            'linf_norm': 0.,
        }

        time_list = []

        while not terminated:

            start = time.time()

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            actions, info, noise = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, epsilon=epsilon)

            if epsilon > 0:
                for k in noise_info.keys():
                    noise_info[k] = max(info[k], noise_info[k])

            if self.args.env in ["particle"]:
                cpu_actions = copy.deepcopy(actions).to("cpu").numpy()
                reward, terminated, env_info = self.env.step(cpu_actions[0])
                if isinstance(reward, (list, tuple)):
                    assert (reward[1:] == reward[:-1]), "reward has to be cooperative!"
                    reward = reward[0]
            else:
                reward, terminated, env_info = self.env.step(actions[0].cpu())
            
            episode_return += reward

            # print(np.shape(self.env.get_obs()), np.shape(noise))

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
                "noise": noise
            }
            # print(post_transition_data['actions'].shape, post_transition_data['reward'], post_transition_data['terminated'], np.shape(noise))

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

            time_list.append(time.time() - start)

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions, info, noise = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, epsilon=epsilon)
        self.batch.update({"actions": actions, "noise": noise}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if epsilon > 0:
            for k in noise_info.keys():
                noise_info[k] = max(info[k], noise_info[k])

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        msg = "{}/{}: {:.2f}, Total Time: {:.2e}, Time step elapse, mean(std): {:.2e} ({:.2e})".format(len(self.test_returns), self.args.test_nepisode, episode_return, np.sum(time_list), np.mean(time_list), np.std(time_list))
        self.logger.print_msg(msg)

        # print(msg)

        # self._log(cur_returns, epsilon, cur_stats, log_prefix)
        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, epsilon, cur_stats, log_prefix, noise_info=noise_info)
        # elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
        #     self._log(cur_returns, epsilon, cur_stats, log_prefix)
            # if self.args.action_selector is not None and hasattr(self.mac.action_selector, "epsilon"):
            #     self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            # self.log_train_stats_t = self.t_env

        return self.batch

    def collect(self, mac=None, test_mode=True):

        if mac is None:
            chosen_mac = self.mac
        else:
            chosen_mac = mac

        self.reset()

        terminated = False
        episode_return = 0
        chosen_mac.init_hidden(batch_size=1)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            actions = chosen_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            # print(self.t, actions, actions.shape, pre_transition_data['obs'])

            if self.args.env in ["particle"]:
                cpu_actions = copy.deepcopy(actions).to("cpu").numpy()
                reward, terminated, env_info = self.env.step(cpu_actions[0])
                if isinstance(reward, (list, tuple)):
                    assert (reward[1:] == reward[:-1]), "reward has to be cooperative!"
                    reward = reward[0]
            else:
                reward, terminated, env_info = self.env.step(actions[0].cpu())
            
            episode_return += reward

            post_transition_data = {
                "actions": actions.cpu().numpy(),
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = chosen_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        # Fix memory leak
        cpu_actions = actions.to("cpu").numpy()
        self.batch.update({"actions": cpu_actions}, ts=self.t)

        return self.batch, self.t

    def _log(self, returns, epsilon, stats, prefix, noise_info=None):

        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        self.logger.log_stat(prefix + "adv_epsilon", epsilon, self.t_env)
        if noise_info is not None:
            self.logger.log_stat(prefix + "epsilon_l1_norm", noise_info['l1_norm'], self.t_env)
            self.logger.log_stat(prefix + "epsilon_l2_norm", noise_info['l2_norm'], self.t_env)
            self.logger.log_stat(prefix + "epsilon_linf_norm", noise_info['linf_norm'], self.t_env)
        print(np.mean(returns), np.std(returns), epsilon)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()

    def dynamic_model_eval(self, test_mode=True):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        total_error = 0
        n_cnt = 0
        running_error = 0
        sm = torch.nn.Softmax(dim=0)

        while not terminated:
            
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            if self.args.env in ["particle"]:
                cpu_actions = copy.deepcopy(actions).to("cpu").numpy()
                reward, terminated, env_info = self.env.step(cpu_actions[0])
                if isinstance(reward, (list, tuple)):
                    assert (reward[1:] == reward[:-1]), "reward has to be cooperative!"
                    reward = reward[0]
            else:
                reward, terminated, env_info = self.env.step(actions[0].cpu())
            
            episode_return += reward

            post_transition_data = {
                "actions": actions.cpu().numpy(),
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1            

            # extract state and probs of corresponding agent
            with torch.no_grad():
                agent_state = agent_inputs[0,self.args.attack_agent].view(1,-1).to(qvals.device)
                probs = sm(qvals[0,self.args.attack_agent]).to(qvals.device)

                agent_state[0,80:91] = probs
                # get predicted state
                next_obs = self.mac.dynamic_model(agent_state)
                next_obs = next_obs.to("cpu").numpy().reshape(-1)


            running_error += np.linalg.norm(np.array(self.env.get_obs()[self.args.attack_agent]).reshape(-1) - next_obs)


            # compute predicted states and error
            if self.t % self.args.model_eval_plan_step == 0:
                total_error += running_error
                running_error = 0
                n_cnt += 1

                # print('real obs', np.array(self.env.get_obs()[self.args.attack_agent]).reshape(-1))
                # print('pred obs', next_obs)

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, epsilon=0)
        # Fix memory leak
        cpu_actions = actions.to("cpu").numpy()
        self.batch.update({"actions": cpu_actions}, ts=self.t)
        
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)
        self.state_est_errors.append(total_error/n_cnt)

        print(test_mode, len(self.test_returns), self.args.test_nepisode, np.mean(self.state_est_errors))

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            # self._log(cur_returns, epsilon, cur_stats, log_prefix)
            self.logger.log_stat(log_prefix + "return_mean", np.mean(cur_returns), self.t_env)
            self.logger.log_stat(log_prefix + "return_std", np.std(cur_returns), self.t_env)
            self.logger.log_stat(log_prefix + "state_est_err", np.mean(self.state_est_errors), self.t_env)
            # self.logger.log_stat(log_prefix + "adv_epsilon", epsilon, self.t_env)
            # print(np.mean(cur_returns), np.std(cur_returns), epsilon)
            cur_returns.clear()

            for k, v in cur_stats.items():
                if k != "n_episodes":
                    self.logger.log_stat(log_prefix + k + "_mean" , v/cur_stats["n_episodes"], self.t_env)
            cur_stats.clear()

        return self.batch