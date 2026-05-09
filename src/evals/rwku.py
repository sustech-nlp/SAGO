from evals.base import Evaluator


class RWKUEvaluator(Evaluator):
    """RWKU evaluation for forget and neighbor sets"""
    
    def __init__(self, eval_cfg, **kwargs):
        super().__init__("RWKU", eval_cfg, **kwargs)
