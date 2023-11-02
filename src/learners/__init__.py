from .maddpg_learner import MADDPGLearner
from .adv_ddpg_learner import AdvDDPGLearner
from .adv_maddpg_learner import AdvMADDPGLearner

REGISTRY = {}

REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["adv_ddpg_learner"] = AdvDDPGLearner
REGISTRY["adv_maddpg_learner"] = AdvMADDPGLearner
