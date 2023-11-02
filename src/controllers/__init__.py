REGISTRY = {}

from .basic_controller import BasicMAC
from .cqmix_controller import CQMixMAC
from .cqmix_adv_noise_controller import CQMixAdvNoiseMAC
from .cqmix_adv_model_controller import CQMixAdvModMAC
from .random_controller import RANDOM_MAC
from .single_basic_controller import SingleBasicMAC
from .single_adv_fgsm_controller import SingleAdvFGSMMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["cqmix_mac"] = CQMixMAC
REGISTRY["cqmix_adv_noise_mac"] = CQMixAdvNoiseMAC
REGISTRY["cqmix_adv_model_mac"] = CQMixAdvModMAC
REGISTRY["random"] = RANDOM_MAC
REGISTRY["single_adv_mac"] = SingleBasicMAC
REGISTRY["single_adv_fgsm_mac"] = SingleAdvFGSMMAC