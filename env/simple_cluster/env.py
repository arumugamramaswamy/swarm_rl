from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn
from gym import spaces

from .scenario import Scenario

import numpy as np


class CustomSimpleEnv(SimpleEnv):
    """
    Main changes here are:
      - observation_spaces modified to return dict_spaces
      - remove the call to astype at the end of observe and state
    """

    def __init__(self, N, scenario, world, max_cycles, continuous_actions, local_ratio):
        super().__init__(
            scenario,
            world,
            max_cycles,
            continuous_actions=continuous_actions,
            local_ratio=local_ratio,
        )

        self.observation_spaces = {}

        for agent in self.possible_agents:
            self.observation_spaces[agent] = spaces.Dict(
                {
                    "my_pos": spaces.Box(-np.inf, np.inf, (world.dim_p,)),
                    "my_vel": spaces.Box(-np.inf, np.inf, (world.dim_p,)),
                    "other_pos": spaces.Box(-np.inf, np.inf, (N - 1, world.dim_p)),
                }
            )

    def render(self, mode="human"):
        from pettingzoo.mpe._mpe_utils import rendering

        if self.viewer is not None and self.render_geoms is not None:
            outer_bound = rendering.make_polygon([(-1, 1), (1,1),(1,-1),(-1,-1)], False)
            outer_xform = rendering.Transform()
            outer_bound.add_attr(outer_xform)
            self.render_geoms.append(outer_bound)
            self.render_geoms_xform.append(outer_xform)

            inner_bound = rendering.make_polygon([(-0.9, 0.9), (0.9,0.9),(0.9,-0.9),(-0.9,-0.9)], False)
            inner_xform = rendering.Transform()
            inner_bound.add_attr(inner_xform)
            self.render_geoms.append(inner_bound)
            self.render_geoms_xform.append(inner_xform)

        super().render(mode)

    def observe(self, agent):
        return self.scenario.observation(
            self.world.agents[self._index_map[agent]], self.world
        )

    def state(self):
        states = tuple(
            self.scenario.observation(
                self.world.agents[self._index_map[agent]], self.world
            )
            for agent in self.possible_agents
        )
        return states


class raw_env(CustomSimpleEnv):
    """
    Main Changes:
      - replace scenario with custom scenario
      - add N to constructor
    """

    def __init__(
        self,
        N=3,
        shuffle=False,
        local_ratio=0.5,
        max_cycles=25,
        continuous_actions=False,
    ):
        assert (
            0.0 <= local_ratio <= 1.0
        ), "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario(
            shuffle=shuffle,
        )
        world = scenario.make_world(N)
        super().__init__(
            N, scenario, world, max_cycles, continuous_actions, local_ratio
        )
        self.metadata["name"] = "simple_cluster"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)
