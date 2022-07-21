import numpy as np
from pettingzoo.mpe.scenarios.simple_spread import Scenario

class CustomScenario(Scenario):
    """The main modification here is of the observation method. It is 
    changed from returning a numpy array to a dict of numpy arrays.

    The reasoning behind this is: having clear information sources will
    allow for better informaiton fusion.
    """
    def __init__(self, shuffle=False) -> None:
        super().__init__()
        self.shuffle=True

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

        entity_pos = np.array(entity_pos, dtype=np.float32)
        other_pos = np.array(other_pos, dtype=np.float32)
        comm = np.array(comm, dtype=np.float32)

        if self.shuffle:
            np.random.shuffle(entity_pos)
            np.random.shuffle(other_pos)
            np.random.shuffle(comm)

        return {
            "my_pos": np.array(agent.state.p_pos, dtype=np.float32),
            "my_vel": np.array(agent.state.p_vel, dtype=np.float32),
            "entity_pos": entity_pos,
            "other_pos": other_pos,
            "comm": comm,
        }
