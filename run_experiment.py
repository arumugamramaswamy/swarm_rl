from parse_config import parse_config, build_viz
from experimenter.core import Experiment

import click

ex = Experiment()


@ex.main
def run_exp(cfg_node, experiment_dir):
    train, test = parse_config(cfg_node, experiment_dir)
    train()
    test()


@ex.group.command(name="viz", help="Visualize model")
@click.argument(
    "cfg_node_path",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    ),
)
@click.argument(
    "model_path",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    ),
)
def viz(cfg_node_path, model_path):
    viz = build_viz(Experiment.load_cfg(cfg_node_path), model_path)
    viz()


ex.run()
