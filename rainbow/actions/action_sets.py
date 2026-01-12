from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DiscreteAction:
    dx: float = 0.0
    dy: float = 0.0
    dtheta: float = 0.0
    shoot: float = 0.0


def baseline_action_set():
    return [
        DiscreteAction(),  # do nothing
        DiscreteAction(dx=-1.0),
        DiscreteAction(dx=1.0),
        DiscreteAction(dy=-1.0),
        DiscreteAction(dy=1.0),
        DiscreteAction(dtheta=-1.0),
        DiscreteAction(dtheta=1.0),
        DiscreteAction(shoot=1.0),
    ]


def compound_action_set():
    return [
        DiscreteAction(),  # 0: idle

        # Cardinal movement
        DiscreteAction(dx=1.0),
        DiscreteAction(dx=-1.0),
        DiscreteAction(dy=1.0),
        DiscreteAction(dy=-1.0),

        # Diagonal movement (0.7 so magnitude is roughly 1)
        DiscreteAction(dx=0.7, dy=0.7),
        DiscreteAction(dx=0.7, dy=-0.7),
        DiscreteAction(dx=-0.7, dy=0.7),
        DiscreteAction(dx=-0.7, dy=-0.7),

        # Rotation
        DiscreteAction(dtheta=0.5),
        DiscreteAction(dtheta=-0.5),

        # Compound movement + rotation
        DiscreteAction(dx=1.0, dtheta=0.3),
        DiscreteAction(dx=-1.0, dtheta=-0.3),

        # Shooting
        DiscreteAction(shoot=1.0),
        DiscreteAction(dx=0.8, shoot=1.0),
        DiscreteAction(dtheta=0.4, shoot=1.0),
    ]
