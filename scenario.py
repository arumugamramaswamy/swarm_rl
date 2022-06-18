import numpy as np
from pettingzoo.mpe.scenarios.simple_spread import Scenario

class CustomScenario(Scenario):
    """The main modification here is of the observation method. It is 
    changed from returning a numpy array to a dict of numpy arrays.

    The reasoning behind this is: having clear information sources will
    allow for better informaiton fusion.
    """
    def observation(self, agent, world):
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return {
            "my_pos": np.array(agent.state.p_pos, dtype=np.float32),
            "my_vel": np.array(agent.state.p_vel, dtype=np.float32),
            "entity_pos": np.array(entity_pos, dtype=np.float32),
            "other_pos": np.array(other_pos, dtype=np.float32),
            "comm": np.array(comm, dtype=np.float32),
        }
