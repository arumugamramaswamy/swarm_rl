from parse_config import parse_config
from experimenter.core import Experiment

ex = Experiment()

@ex.main
def run_exp(cfg_node, experiment_dir):
    train, test = parse_config(cfg_node, experiment_dir)
    train()
    test()

ex.run()
