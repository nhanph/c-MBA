from math import sqrt
import numpy as np

from envs.multiagentenv import MultiAgentEnv
# from envs.particle.environment import MultiAgentEnv as OpenAIMultiAgentEnv
from multiagent.environment import MultiAgentEnv as OpenAIMultiAgentEnv
from envs.particle import scenarios
from gym import spaces

# def make_env(scenario_name="simple_tag", benchmark=False, **kwargs):
#     '''
#     Creates a MultiAgentEnv object as env. This can be used similar to a gym
#     environment by calling env.reset() and env.step().
#     Use env.render() to view the environment on the screen.
#     Input:
#         scenario_name   :   name of the scenario from ./scenarios/ to be Returns
#                             (without the .py extension)
#         benchmark       :   whether you want to produce benchmarking data
#                             (usually only done during evaluation)
#     Some useful env properties (see environment.py):
#         .observation_space  :   Returns the observation space for each agent
#         .action_space       :   Returns the action space for each agent
#         .n                  :   Returns the number of Agents
#     '''
#     from multiagent.environment import MultiAgentEnv as MultiAgentEnv2
#     import multiagent.scenarios as scenarios

#     # load scenario from script
#     scenario = scenarios.load(scenario_name + ".py").Scenario()
#     # create world
#     world = scenario.make_world()
#     # create multiagent environment
#     if benchmark:        
#         env = MultiAgentEnv2(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
#     else:
#         env = MultiAgentEnv2(world, scenario.reset_world, scenario.reward, scenario.observation)
#     return env

class Particle(MultiAgentEnv):

    def __init__(self, batch_size=None, **kwargs):
        super().__init__(batch_size, **kwargs)

        # load scenario from script
        self.episode_limit = self.args.episode_limit
        self.scenario = scenarios.load(self.args.scenario + ".py").Scenario()
        if not self.args.partial_obs:
            self.world = self.scenario.make_world()
        else:
            self.world = self.scenario.make_world(self.args)
        self.n_agents = len(self.world.policy_agents)
        self.steps = 0
        self.truncate_episodes = getattr(self.args, "truncate_episodes", True)
        self.total_steps = 0

        if self.args.benchmark:
            self.env = OpenAIMultiAgentEnv(self.world,
                                            self.scenario.reset_world,
                                            self.scenario.reward,
                                            self.scenario.observation,
                                            self.scenario.benchmark_data)
        else:
            if not self.args.partial_obs:
                self.env = OpenAIMultiAgentEnv(self.world,
                                                self.scenario.reset_world,
                                                self.scenario.reward,
                                                self.scenario.observation)
            else:
                self.env = OpenAIMultiAgentEnv(self.world,
                                               self.scenario.reset_world,
                                               self.scenario.reward,
                                               self.scenario.observation)

        self.glob_args = kwargs.get("args")

    def step(self, actions):
        obs_n, reward_n, done_n, info_n = self.env.step(actions)

        self.steps += 1
        self.is_done = all(done_n)
        terminated = all(done_n)

        if self.steps >= self.episode_limit and not terminated:
            terminated = True
            info_n["episode_limit"] = getattr(self, "truncate_episodes", True)  # by default True
        else:
            info_n["episode_limit"] = False

        # test minimum distance to a landmark
        min_dists = []
        for agent in self.world.agents:
            min_dists.append(float("inf"))
            for landmark in self.world.landmarks:
                dist = sqrt(sum((apos-lpos)**2 for apos, lpos in zip(agent.state.p_pos, landmark.state.p_pos)))
                if dist < min_dists[-1]:
                    min_dists[-1] = dist

        info_n["min_dists_mean"] = np.mean(min_dists)
        if hasattr(self.scenario, "n_last_collisions"):
            info_n["n_last_collisions"] = self.scenario.n_last_collisions

        if "n" in info_n:
            del info_n["n"]

        for i, min_dist in enumerate(min_dists):
            info_n["mind_dist__agent{}".format(i)] = min_dist
        return reward_n, terminated, info_n

    def get_obs(self):
        """ Returns all agent observations in a list """
        obs_n = []
        for i, _ in enumerate(self.world.policy_agents):
            obs = self.get_obs_agent(i)
            obs_n.append(obs)
        return obs_n

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        obs = self.env._get_obs(self.world.policy_agents[agent_id])
        if len(obs) < self.get_obs_size():
            obs = np.concatenate([obs, np.zeros((self.get_obs_size() - len(obs)))],
                                 axis=0)  # pad all obs to same length
        return obs

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return max([o.shape[0] for o in self.env.observation_space])

    def get_state(self, team=None):
        state = np.concatenate(self.get_obs())
        return state

    def get_state_size(self):
        """ Returns the shape of the state"""
        state_size = len(self.get_state())
        return state_size

    def get_avail_actions(self):
        return np.ones((self.n_agents, self.get_total_actions()))

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        raise NotImplementedError

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # return max([x.n for x in self.env.action_space])
        if all([isinstance(act_space, spaces.Discrete) for act_space in self.env.action_space]):
            return max([x.n for x in self.env.action_space])
        elif all([isinstance(act_space, spaces.Box) for act_space in self.env.action_space]):
            if self.args.scenario == "simple_speaker_listener":
                return self.env.action_space[0].shape[0] + self.env.action_space[1].shape[0]
            else:
                return max([x.shape[0] for x in self.env.action_space])
        elif all([isinstance(act_space, spaces.Tuple) for act_space in self.env.action_space]):
            return max([x.spaces[0].shape[0] + x.spaces[1].shape[0] for x in self.env.action_space])
        else:
            raise Exception("not implemented for this scenario!")

    def get_stats(self):
        return {}

    def get_agg_stats(self, stats):
        return {}

    def reset(self, force_reset=False):
        """ Returns initial observations and states"""
        self.total_steps += self.steps
        self.steps = 0
        if (not getattr(self.glob_args, "continuous_episode", False)) or force_reset or self.is_done:
            self.is_done = False
            self.env.reset()
        pass

    def render(self, **kwargs):
        self.env.render(**kwargs)

    def close(self):
        raise NotImplementedError

    def seed(self):
        raise NotImplementedError

    def get_env_info(self):
        action_spaces = self.env.action_space

        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit,
                    "action_spaces": action_spaces,
                    "actions_dtype": np.float32,
                    "normalise_actions": False}
        return env_info