from .run import run as default_run
from .eval_run import run as eval_run
from .single_run import run as single_run

REGISTRY = {}
REGISTRY["default"] = default_run
REGISTRY["eval_run"] = eval_run
REGISTRY["single_run"] = single_run