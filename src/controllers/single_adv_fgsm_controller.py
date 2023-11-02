from gym import spaces
import torch as th
import torch.distributions as tdist
import numpy as np

from .basic_controller import BasicMAC
from modules.agents import REGISTRY as agent_REGISTRY
from utils.rl_utils import project_l1_ball, project_l2_ball
import os


# This multi-agent controller shares parameters between agents
class SingleAdvFGSMMAC(BasicMAC):

    def __init__(self, scheme, groups, args):

        super(SingleAdvFGSMMAC, self).__init__(scheme, groups, args)

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

        input_shape = self._get_input_shape(scheme)

        if hasattr(args,'single_adv_model_path'):
            if args.single_adv_model_path != "":

                model_paths=[]

                for agent in self.attack_agent:

                    timesteps = []
                    timestep_to_load = 0

                    # Go through all files in args.single_adv_model_path
                    full_path = args.single_adv_model_path + '_agent' + str(agent)
                    # print('aaa',args.single_adv_model_path + '_agent' + str(agent))
                    for name in os.listdir(full_path):
                        full_name = os.path.join(full_path, name)
                        # Check if they are dirs the names of which are numbers
                        if os.path.isdir(full_name) and name.isdigit():
                            timesteps.append(int(name))

                    if args.load_step == 0:
                        # choose the max timestep
                        timestep_to_load = max(timesteps)
                    else:
                        # choose the timestep closest to load_step
                        timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

                    model_path = os.path.join(full_path, str(timestep_to_load))
                    model_paths.append(model_path)
                
                self.init_adv_agent(input_shape, model_paths)

        else:
            raise ValueError('adv model path not found')

    def init_adv_agent(self, input_shape, paths):
        self.adv_agent_dict = {}

        # print(len(paths),paths)
        for idx, agent in enumerate(self.attack_agent):
            self.adv_agent_dict[agent] = agent_REGISTRY[self.args.agent](input_shape, self.args).to(self.args.device)
            # load trained model
            if self.args.fix_adv_policy:
                self.adv_agent_dict[agent].load_state_dict(th.load("{}/agent.th".format(paths[0]), map_location=lambda storage, loc: storage))
            else:
                self.adv_agent_dict[agent].load_state_dict(th.load("{}/agent.th".format(paths[idx]), map_location=lambda storage, loc: storage))

            for param in self.adv_agent_dict[agent].parameters():
                param.requires_grad = False


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

        final_noise = th.zeros_like(agent_inputs).to(agent_inputs.device)

        if epsilon == 0:
            noise_info = {
                "l1_norm": 0.,
                "l2_norm": 0.,
                "linf_norm": 0.,
            }
        else:
            loss_fn = th.nn.MSELoss()

            # select attack agent
            if type(self.attack_agent) == list:
                if len(self.attack_agent) <= self.args.num_atk_agent:
                    agents = self.attack_agent
                else:
                    # random from a set
                    agents = np.random.choice(self.attack_agent, self.args.num_atk_agent, replace=False)
            else:
                raise ValueError('Unsupported type of attack_agent',type(self.attack_agent))

            noise_info = {
                    "l1_norm": 0.,
                    "l2_norm": 0.,
                    "linf_norm": 0.,
                }

            for agent_idx in agents:
                original_inputs = agent_inputs[agent_idx]
                new_inputs = original_inputs.clone().detach()
                new_inputs.requires_grad=True

                target_actions = self.adv_agent_dict[agent_idx](new_inputs.view(1,-1), actions=actions)
                target_actions = target_actions['actions'].view(1,-1)
                # print('bbb',target_actions.shape)

                for fi in range(self.args.fgsm_iter):
                    # print('iter',fi, new_inputs)
                    agent_outs = self.agent(new_inputs.view(1,-1), actions=None)['actions']

                    # print(output.shape, target.shape)
                    loss = loss_fn(agent_outs, target_actions.detach())
                    self.agent.zero_grad()
                    loss.backward()

                    # collect datagrad
                    data_grad = new_inputs.grad.data
                    with th.no_grad():
                        # call FSGM attack
                        perturbed_data = self.fgsm_attack(new_inputs, data_grad, lr=self.args.fgsm_lr)

                    new_inputs = perturbed_data.clone().detach()
                    new_inputs.requires_grad=True

                with th.no_grad():
                    adv_noise = new_inputs - original_inputs
                    if self.args.noise_constraint_type == 'l1_norm':
                        adv_noise = project_l1_ball(adv_noise, rad=epsilon)
                    elif self.args.noise_constraint_type == 'l2_norm':
                        adv_noise = project_l2_ball(adv_noise, rad=epsilon)
                    else:
                        adv_noise = adv_noise.clamp(-epsilon,epsilon)

                with th.no_grad():
                    noise_info = {
                        "l1_norm": th.linalg.norm(adv_noise.view(-1), ord=1).item(),
                        "l2_norm": th.linalg.norm(adv_noise.view(-1)).item(),
                        "linf_norm": th.linalg.norm(adv_noise.view(-1), ord=float('inf')).item(),
                    }

                    agent_inputs[agent_idx] = original_inputs.view(1,-1) + adv_noise

                final_noise[agent_idx] = adv_noise.clone().detach()

        ret = self.agent(agent_inputs, actions=None)

        # print('ddd',ret['actions'].shape, agent_inputs.shape)

        return ret['actions'], noise_info, final_noise.to('cpu').clone().detach().numpy().tolist()


    def fgsm_attack(self, input_data, data_grad, lr=0.01):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed input_data by adjusting each pixel of the input data
        perturbed_input = input_data + lr*sign_data_grad
 
        return perturbed_input

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])

        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions"][:, t]))
            else:
                inputs.append(batch["actions"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)

        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    def cuda(self, device="cuda"):
        self.agent.cuda(device=device)
        for k in self.adv_agent_dict.keys():
            self.adv_agent_dict[k].cuda(device=device)    
        
    def parameters(self):
        return self.agent.parameters()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def load_trained_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))