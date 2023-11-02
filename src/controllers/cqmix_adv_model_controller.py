from gym import spaces
import torch
import torch.distributions as tdist
import numpy as np

from .basic_controller import BasicMAC

from modules.dynamic_model import REGISTRY as model_REGISTRY
from modules.dynamic_model.models import FCNetWithSoftmax
from utils.rl_utils import project_l1_ball, project_l2_ball
from itertools import combinations

# This multi-agent controller shares parameters between agents
class CQMixAdvModMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(CQMixAdvModMAC, self).__init__(scheme, groups, args)

        if hasattr(args,'attack_agent'):
            if args.attack_agent is not None:
                if type(args.attack_agent) != list:
                    self.attack_agent = [args.attack_agent]
                else:
                    self.attack_agent = args.attack_agent
            else:
                raise ValueError('Need to specify attack_agent')
        else:
            raise ValueError('Need to specify attack_agent')

        self.init_dynamic_model(args)
        self.action_device = None

        self.agent.eval()
        self.agent.requires_grad_(False)

        if self.args.optimal_adv_agent:
            self.opt_adv_model = FCNetWithSoftmax(n_feature=self.obs_shape, n_output=1, n_hidden=(200,200), sm_temp=1.)
            self.opt_adv_model = self.opt_adv_model.to(self.args.device)

    def init_dynamic_model(self, args):
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.obs_shape = args.obs_shape

        self.n_features = self.n_agents + self.n_actions + self.obs_shape
        self.n_output = self.obs_shape

        self.dynamic_model = model_REGISTRY[args.dynamic_model](n_feature=self.n_features, n_output=self.n_output)

        # load dynamic model and move to GPU
        try:
            self.load_dynamic_model(args.dynamic_model_path)
            self.dynamic_model.eval()
            self.dynamic_model.requires_grad_(False)
        except:
            print('Exception')

    def load_dynamic_model(self, model_path):
        self.dynamic_model.load_state_dict(torch.load(model_path))
        self.dynamic_model = self.dynamic_model.to(self.args.device)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, epsilon=0.):
          
        chosen_actions, info, final_noise = self.forward(ep_batch[bs], t_ep, test_mode=test_mode, select_actions=True, epsilon=epsilon)
        chosen_actions = chosen_actions.view(ep_batch[bs].batch_size, self.n_agents, self.args.n_actions).detach()

        # now clamp actions to permissible action range (necessary after exploration)
        if all([isinstance(act_space, spaces.Box) for act_space in self.args.action_spaces]):
            for _aid in range(self.n_agents):
                for _actid in range(self.args.action_spaces[_aid].shape[0]):
                    chosen_actions[:, _aid, _actid].clamp_(np.asscalar(self.args.action_spaces[_aid].low[_actid]),
                                                           np.asscalar(self.args.action_spaces[_aid].high[_actid]))
        elif all([isinstance(act_space, spaces.Tuple) for act_space in self.args.action_spaces]):
            for _aid in range(self.n_agents):
                for _actid in range(self.args.action_spaces[_aid].spaces[0].shape[0]):
                    chosen_actions[:, _aid, _actid].clamp_(self.args.action_spaces[_aid].spaces[0].low[_actid],
                                                           self.args.action_spaces[_aid].spaces[0].high[_actid])
                for _actid in range(self.args.action_spaces[_aid].spaces[1].shape[0]):
                    tmp_idx = _actid + self.args.action_spaces[_aid].spaces[0].shape[0]
                    chosen_actions[:, _aid, tmp_idx].clamp_(self.args.action_spaces[_aid].spaces[1].low[_actid],
                                                            self.args.action_spaces[_aid].spaces[1].high[_actid])
        return chosen_actions, info, final_noise

    def get_weight_decay_weights(self):
        return self.agent.get_weight_decay_weights()

    def forward(self, ep_batch, t, actions=None, select_actions=False, test_mode=False, epsilon=0.):
        agent_inputs = self._build_inputs(ep_batch, t)

        final_noise = torch.zeros_like(agent_inputs).to(agent_inputs.device)

        if epsilon > 0:

            if self.args.optimal_adv_agent:
                if self.args.brute_force_atk:
                    input_noise = self.opt_model_atk_brute_force(agent_inputs, t, epsilon=epsilon)
                else:
                    input_noise = self.opt_model_atk(agent_inputs, t, epsilon=epsilon)

            else:
                ## perform attack
                # select attack agent
                if type(self.attack_agent) == list:
                    if len(self.attack_agent) <= self.args.num_atk_agent:
                        agents = self.attack_agent
                    else:
                        # random from a set
                        agents = np.random.choice(self.attack_agent, self.args.num_atk_agent, replace=False)
                else:
                    raise ValueError('Unsupported type of attack_agent',type(self.attack_agent))

                input_noise = self.model_atk(agent_inputs, agents, t, epsilon=epsilon)

            with torch.no_grad():
                agent_inputs += input_noise

            noise_info = {
                "l1_norm": torch.max(torch.sum(torch.abs(final_noise.view(self.n_agents,-1)), 1)).item(),
                "l2_norm": torch.max(torch.sqrt(torch.sum(torch.square(final_noise.view(self.n_agents,-1)), 1))).item(),
                "linf_norm": torch.max(torch.abs(final_noise.view(self.n_agents,-1))).item()
            }

            final_noise = input_noise
        else:
            noise_info = {
                # "value": final_noise.to('cpu'),
                "l1_norm": 0,
                "l2_norm": 0,
                "linf_norm": 0,
            }

        ret = self.agent(agent_inputs, actions=actions)

        return ret["actions"], noise_info, final_noise.to('cpu').clone().detach().numpy().tolist()

    def model_atk(self, agent_inputs, agent_idx, t, epsilon=0.):
        loss_func = torch.nn.MSELoss()

        final_noise = torch.zeros_like(agent_inputs).to(agent_inputs.device)
        input_noise = torch.zeros_like(agent_inputs).to(agent_inputs.device)

        # only work when batch_size_run = 1
        new_inputs = agent_inputs.clone().detach()

        if hasattr(self.args,'init_type'):
            if self.args.init_type == 'random':
                input_noise[agent_idx] = epsilon*(2*torch.rand(new_inputs[agent_idx].shape, device=agent_inputs.device)-1.)
            elif self.args.init_type == 'epsilon':
                input_noise[agent_idx] = epsilon*torch.sign(torch.randn(new_inputs[agent_idx].shape, device=agent_inputs.device))
            elif self.args.init_type == 'zero':
                input_noise[agent_idx] = torch.zeros(new_inputs[agent_idx].shape, device=agent_inputs.device)
            else:
                input_noise[agent_idx] = self.args.init_val*torch.ones(new_inputs[agent_idx].shape, device=agent_inputs.device)

        input_noise.requires_grad=True

        if self.args.pgd_solver == 'adam':
            optimizer = torch.optim.Adam([input_noise], lr=self.args.pgd_lr)
        else:
            optimizer = torch.optim.SGD([input_noise], lr=self.args.pgd_lr)

        agent_onehot = torch.eye(self.args.n_agents).to(agent_inputs.device)

        for step in range(self.args.pgd_step):

            # forward pass
            loss = 0
            for p_step in range(self.args.plan_step):
                # update new obs
                if p_step == 0:
                    cat_ts = []
                    for i in range(self.args.n_agents):
                        if i in agent_idx:
                            cat_ts.append((new_inputs[i] + input_noise[i]).view(1,-1))
                        else:
                            cat_ts.append(new_inputs[i].view(1,-1))
                    # perturbed_inputs = new_inputs + input_noise
                    perturbed_inputs = torch.cat(cat_ts, dim=0)
                else:
                    perturbed_inputs = new_inputs.clone().detach()

                # get actions
                action_outs = self.agent(perturbed_inputs)["actions"]

                # concatenate to form input to dynamic model
                new_ts = torch.cat((new_inputs[:,:self.n_output],
                                    action_outs,
                                    agent_onehot),dim=-1)

                # get next obs via dynamic model
                next_obs = self.dynamic_model(new_ts)
                atk_obs = next_obs[agent_idx]

                # determine desired obs
                desired_obs = self.build_desire_obs(atk_obs, agent_idx)
                loss += loss_func(desired_obs,atk_obs)

            # zero grad
            optimizer.zero_grad()

            # backward pass
            loss.backward()

            # update perturbation
            optimizer.step()
            
            # with torch.no_grad():
            if self.args.noise_constraint_type == 'l1_norm':
                for agent in agent_idx:
                    input_noise.data[agent] = project_l1_ball(input_noise[agent], rad=epsilon)
            elif self.args.noise_constraint_type == 'l2_norm':
                for agent in agent_idx:
                    input_noise.data[agent] = project_l2_ball(input_noise[agent], rad=epsilon)
            else:
                input_noise.data = input_noise.clamp(-epsilon,epsilon)

        final_noise = input_noise.detach()

        return final_noise

    def opt_model_atk(self, agent_inputs, t, epsilon=0.):
        loss_func = torch.nn.MSELoss()

        final_noise = torch.zeros_like(agent_inputs).to(agent_inputs.device)

        # only work when batch_size_run = 1
        new_inputs = agent_inputs.clone().detach()
        if hasattr(self.args,'init_type'):
            if self.args.init_type == 'random':
                input_noise = epsilon*(2*torch.rand(new_inputs.shape, device=agent_inputs.device)-1.)
            elif self.args.init_type == 'epsilon':
                input_noise = epsilon*torch.sign(torch.randn(new_inputs.shape, device=agent_inputs.device))
            elif self.args.init_type == 'zero':
                input_noise = torch.zeros(new_inputs.shape, device=agent_inputs.device)
            else:
                input_noise = self.args.init_val*torch.ones(new_inputs.shape, device=agent_inputs.device)
        else:            
            input_noise = torch.zeros(new_inputs.shape, device=agent_inputs.device)

        input_noise.requires_grad=True

        if self.args.pgd_solver == 'adam':
            optimizer = torch.optim.Adam([input_noise], lr=self.args.pgd_lr)
            nw_optimizer = torch.optim.Adam(self.opt_adv_model.parameters(), lr=self.args.pgd_lr)
        else:
            optimizer = torch.optim.SGD([input_noise], lr=self.args.pgd_lr)
            nw_optimizer = torch.optim.SGD(self.opt_adv_model.parameters(), lr=self.args.pgd_lr)

        agent_onehot = torch.eye(self.args.n_agents).to(agent_inputs.device)

        for step in range(self.args.pgd_step):
            # forward pass
            loss = 0
            for p_step in range(self.args.plan_step):
                # update new obs
                if p_step == 0:
                    # get noise weight
                    noise_weight = self.opt_adv_model(new_inputs)

                    # perturbed_inputs = new_inputs + input_noise
                    perturbed_inputs = new_inputs + input_noise*noise_weight

                    # print(noise_weight)

                else:
                    perturbed_inputs = new_inputs.clone().detach()

                # get q-values
                action_outs = self.agent(perturbed_inputs)["actions"]

                # concatenate to form input to dynamic model
                new_ts = torch.cat((new_inputs[:,:self.n_output],
                                    action_outs,
                                    agent_onehot),dim=-1)

                # get next obs via dynamic model
                next_obs = self.dynamic_model(new_ts)
                # atk_obs = next_obs[agent_idx]

                # determine desired obs
                desired_obs = self.build_desire_obs(next_obs, agent_idx)
                loss += loss_func(desired_obs,next_obs)

            # zero grad
            optimizer.zero_grad()
            nw_optimizer.zero_grad()

            # backward pass
            loss.backward()

            # update noise
            optimizer.step()
            nw_optimizer.step()
            
            # with torch.no_grad():
            if self.args.noise_constraint_type == 'l1_norm':
                for i in range(self.n_agents):
                    input_noise.data[i] = project_l1_ball(input_noise[i], rad=epsilon)
            elif self.args.noise_constraint_type == 'l2_norm':
                for i in range(self.n_agents):
                    input_noise.data[i] = project_l2_ball(input_noise[i], rad=epsilon)
            else:
                input_noise.data = input_noise.clamp(-epsilon,epsilon)


        final_weight = self.opt_adv_model(agent_inputs).view(-1)
        _, indices = torch.topk(final_weight.view(-1), self.args.num_atk_agent)

        if self.args.attack_again:
            final_noise = self.model_atk(agent_inputs, indices, t, epsilon=epsilon)

        else:
            final_noise[indices] = input_noise[indices].clone().detach()

        return final_noise

    def opt_model_atk_brute_force(self, agent_inputs, t, epsilon=0.):
        loss_func = torch.nn.MSELoss()

        # only work when batch_size_run = 1
        new_inputs = agent_inputs.clone().detach()

        final_noise = torch.zeros_like(agent_inputs).to(agent_inputs.device)
        tmp_noise = torch.zeros_like(agent_inputs).to(agent_inputs.device)
        losses = torch.zeros(self.args.n_agents).to(agent_inputs.device)

        if hasattr(self.args,'init_type'):
            if self.args.init_type == 'random':
                input_noise = epsilon*(2*torch.rand(new_inputs.shape, device=agent_inputs.device)-1.)
            elif self.args.init_type == 'epsilon':
                input_noise = epsilon*torch.sign(torch.randn(new_inputs.shape, device=agent_inputs.device))
            elif self.args.init_type == 'zero':
                input_noise = torch.zeros(new_inputs.shape, device=agent_inputs.device)
            else:
                input_noise = self.args.init_val*torch.ones(new_inputs.shape, device=agent_inputs.device)
        else:            
            input_noise = torch.zeros(new_inputs.shape, device=agent_inputs.device)

        agent_onehot = torch.eye(self.args.n_agents).to(agent_inputs.device)

        # perform attack on each agent
        for agent_idx in range(self.n_agents):
        # for agent_idx in range(self.args.n_agents):

            noise_copy = input_noise.clone().detach()
            noise_copy.requires_grad=True
            # input_noise.requires_grad=True
            if self.args.pgd_solver == 'adam':
                optimizer = torch.optim.Adam([noise_copy], lr=self.args.pgd_lr)
            else:
                optimizer = torch.optim.SGD([noise_copy], lr=self.args.pgd_lr)

            for step in range(self.args.pgd_step):
                # print('PGD step', step)

                # forward pass
                loss = 0
                for p_step in range(self.args.plan_step):
                    # update new obs
                    if p_step == 0:
                        # perturb only the attacked agents
                        perturbed_inputs = torch.cat((new_inputs[:agent_idx], 
                                      new_inputs[agent_idx:agent_idx+1] + noise_copy[agent_idx:agent_idx+1], 
                                      new_inputs[agent_idx+1:]), dim=0)
                    else:
                        perturbed_inputs = new_inputs.clone().detach()

                    # get q-values
                    action_outs = self.agent(perturbed_inputs)["actions"]

                    # concatenate to form input to dynamic model
                    new_ts = torch.cat((new_inputs[:,:self.n_output],
                                        action_outs,
                                        agent_onehot),dim=-1)

                    # get next obs via dynamic model
                    next_obs = self.dynamic_model(new_ts)
                    atk_obs = next_obs[agent_idx]

                    # determine desired obs
                    desired_obs = self.build_desire_obs(atk_obs, agent_idx)
                    loss += loss_func(desired_obs,atk_obs)

                # zero grad
                optimizer.zero_grad()

                # backward pass
                loss.backward()

                # update noise
                optimizer.step()
                
                # project to norm constraint set
                if self.args.noise_constraint_type == 'l1_norm':
                    noise_copy.data[agent_idx] = project_l1_ball(noise_copy[agent_idx], rad=epsilon)
                elif self.args.noise_constraint_type == 'l2_norm':
                    noise_copy.data[agent_idx] = project_l2_ball(noise_copy[agent_idx], rad=epsilon)
                else:
                    noise_copy.data[agent_idx] = noise_copy[agent_idx].clamp(-epsilon,epsilon)

            # update list
            tmp_noise[agent_idx] = noise_copy[agent_idx].clone().detach()
            losses[agent_idx] = -loss.item()

        # get top k of negative losses = get min k
        _, indices = torch.topk(losses, self.args.num_atk_agent)

        final_noise[indices] = tmp_noise[indices]

        return final_noise

    def build_desire_obs(self, current_obs, agent_idx):

        if type(agent_idx) == list:
            agent_idx = agent_idx[0]

        if current_obs.ndim > 1:
            with torch.no_grad():
                if self.args.env_args['scenario'] == 'Walker2d-v2':
                    desire_obs = current_obs.clone()
                    if self.args.data_driven_failure_state == False:
                        desire_obs[:,0] = 0
                    else:
                        # defined from collected data
                        failure_state = torch.tensor([  1.1586,  -0.5691,  -0.9223,  -0.0815,   0.8496,  -0.3560,  -0.0238,
                                                    0.8281,  -3.2239,  -0.6004, -10.0000,   2.5125, -10.0000,  -0.9436,
                                                    -3.3814, -10.0000,   0.0982])
                        desire_obs[:] = failure_state
                elif self.args.env_args['scenario'] == 'HalfCheetah-v2':
                    desire_obs = current_obs.clone()
                    if self.args.data_driven_failure_state == False:
                        desire_obs[:,8] = 0
                    else:
                        if self.args.env_args['agent_conf'] == "2x3":
                            # defined from collected data
                            failure_state = torch.tensor([-4.6806e-01,  3.3428e+00,  2.5579e-02, -7.6699e-01, -5.1423e-01,
                                            4.6807e-02, -6.5985e-01, -3.8235e-01, -3.3638e+00, -6.1622e-01,
                                            -3.9170e+00,  1.3512e+01, -1.4710e+00,  2.9961e+00,  2.5856e+01,
                                            -3.5981e+00, -9.6768e+00], device=current_obs.device)
                        else:
                            # defined from collected data
                            failure_state = torch.tensor([ -0.0283,   6.2424,  -0.0562,  -0.2495,  -0.2651,  -0.0886,   0.0187,
                                            -0.0370,  -5.3303,  -1.0764,   0.4602,  -8.1812, -11.3249, -11.7932,
                                            -15.8474,  -9.2553,  -9.7467], device=current_obs.device)
                        desire_obs[:] = failure_state
                elif self.args.env_args['scenario'] == 'Ant-v2':
                    desire_obs = current_obs.clone()
                    if self.args.data_driven_failure_state == False:
                        desire_obs[:,13:14] = 0
                    else:
                        # defined from collected data
                        failure_state = torch.tensor([ 0.6597,  0.9989, -0.0263,  0.0349, -0.0161,  0.3369,  1.0918,  0.3894,
                                            -1.0523,  0.2911, -1.0603,  0.3706,  1.1016, -1.7878,  0.4393, -0.1900,
                                            -0.2478, -1.2498,  3.3063, -7.5145, -2.1409, -6.2961, -2.0204, -8.0260,
                                            -2.3870, -6.6400, -6.7671,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                                            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                                            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                                            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                                            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                                            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                                            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                                            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                                            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                                            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                                            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000], device=current_obs.device)
                        desire_obs[:] = failure_state

                elif self.args.env_args['scenario'] == 'simple_tag_coop':
                    # no expert knowledge, use data driven one
                    desire_obs = current_obs.clone()
                    if agent_idx == 0:
                        # defined from collected data
                        failure_state = torch.tensor([ 0.4191,  0.0878,  1.1662,  0.7827, -0.3318, -1.4324,  0.0000,  0.0000,
                                                    -0.0578, -0.1279,  0.3236, -0.3036,  0.1251, -0.2598,  0.0557, -0.2078], device=current_obs.device)
                    elif agent_idx == 1:
                        # defined from collected data
                        failure_state = torch.tensor([ 0.3000,  0.3000, -0.4173, -0.7347, -0.1813,  0.4931,  1.1313,  0.9094,
                                                     0.0026,  0.6205, -0.3886,  0.5779,  0.6957,  0.6605,  0.0943,  0.0230], device=current_obs.device)
                    elif agent_idx == 2:
                        # defined from collected data
                        failure_state = torch.tensor([ 0.1709,  0.2752,  0.6816,  1.3039,  0.0000,  0.0000, -0.5030, -1.3336,
                                                     0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000], device=current_obs.device)

                    desire_obs[:] = failure_state
                else:
                    raise ValueError('Scenario {} not supported'.format(self.args.env_args['scenario']))

                return desire_obs
        else:
            with torch.no_grad():
                if self.args.env_args['scenario'] == 'Walker2d-v2':
                    if self.args.data_driven_failure_state == False:
                        desire_obs = current_obs.clone().reshape(-1)
                        desire_obs[0] = 0
                    else:
                        # defined from collected data
                        desire_obs = torch.tensor([  1.1586,  -0.5691,  -0.9223,  -0.0815,   0.8496,  -0.3560,  -0.0238,
                                                    0.8281,  -3.2239,  -0.6004, -10.0000,   2.5125, -10.0000,  -0.9436,
                                                    -3.3814, -10.0000,   0.0982])
                elif self.args.env_args['scenario'] == 'HalfCheetah-v2':
                    if self.args.env_args['agent_conf'] == "2x3":
                        if self.args.data_driven_failure_state == False:
                            desire_obs = current_obs.clone().reshape(-1)
                            desire_obs[8] = 0
                        else:
                            desire_obs = torch.tensor([-4.6806e-01,  3.3428e+00,  2.5579e-02, -7.6699e-01, -5.1423e-01,
                                            4.6807e-02, -6.5985e-01, -3.8235e-01, -3.3638e+00, -6.1622e-01,
                                            -3.9170e+00,  1.3512e+01, -1.4710e+00,  2.9961e+00,  2.5856e+01,
                                            -3.5981e+00, -9.6768e+00], device=current_obs.device)
                    else:
                        if self.args.data_driven_failure_state == False:
                            desire_obs = current_obs.clone().reshape(-1)
                            desire_obs[8] = 0
                        else:
                            desire_obs = torch.tensor([ -0.0283,   6.2424,  -0.0562,  -0.2495,  -0.2651,  -0.0886,   0.0187,
                                            -0.0370,  -5.3303,  -1.0764,   0.4602,  -8.1812, -11.3249, -11.7932,
                                            -15.8474,  -9.2553,  -9.7467], device=current_obs.device)

                elif self.args.env_args['scenario'] == 'Ant-v2':
                    if self.args.data_driven_failure_state == False:
                        desire_obs = current_obs.clone().reshape(-1)
                        desire_obs[13:15] = 0
                    else:
                        desire_obs = torch.tensor([ 0.6597,  0.9989, -0.0263,  0.0349, -0.0161,  0.3369,  1.0918,  0.3894,
                                            -1.0523,  0.2911, -1.0603,  0.3706,  1.1016, -1.7878,  0.4393, -0.1900,
                                            -0.2478, -1.2498,  3.3063, -7.5145, -2.1409, -6.2961, -2.0204, -8.0260,
                                            -2.3870, -6.6400, -6.7671,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                                            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                                            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                                            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                                            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                                            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                                            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                                            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                                            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                                            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                                            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000], device=current_obs.device)
                    
                elif self.args.env_args['scenario'] == 'simple_tag_coop':
                    desire_obs = current_obs.clone().reshape(-1)
                    if agent_idx == 0:
                        desire_obs = torch.tensor([ 0.4191,  0.0878,  1.1662,  0.7827, -0.3318, -1.4324,  0.0000,  0.0000,
                                                    -0.0578, -0.1279,  0.3236, -0.3036,  0.1251, -0.2598,  0.0557, -0.2078], device=current_obs.device)
                    elif agent_idx == 1:
                        desire_obs = torch.tensor([ 0.3000,  0.3000, -0.4173, -0.7347, -0.1813,  0.4931,  1.1313,  0.9094,
                                                     0.0026,  0.6205, -0.3886,  0.5779,  0.6957,  0.6605,  0.0943,  0.0230], device=current_obs.device)
                    elif agent_idx == 2:
                        desire_obs = torch.tensor([ 0.1709,  0.2752,  0.6816,  1.3039,  0.0000,  0.0000, -0.5030, -1.3336,
                                                     0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000], device=current_obs.device)
                else:
                    raise ValueError('Scenario {} not supported'.format(self.args.env_args['scenario']))

                return desire_obs.reshape(current_obs.shape)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])

        if self.args.obs_last_action:
            if t == 0:
                inputs.append(torch.zeros_like(batch["actions"][:, t]))
            else:
                inputs.append(batch["actions"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(torch.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = torch.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)

        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape