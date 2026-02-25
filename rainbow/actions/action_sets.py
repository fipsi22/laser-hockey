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
        DiscreteAction(),  # idle

        # Cardinal movement
        DiscreteAction(dx=1.0),
        DiscreteAction(dx=-1.0),
        DiscreteAction(dy=1.0),
        DiscreteAction(dy=-1.0),

        # Diagonal (normalized)
        DiscreteAction(dx=0.707, dy=0.707),
        DiscreteAction(dx=0.707, dy=-0.707),
        DiscreteAction(dx=-0.707, dy=0.707),
        DiscreteAction(dx=-0.707, dy=-0.707),

        # Diagonal (normalized) rotation
        DiscreteAction(dx=0.707, dy=0.707, dtheta=1.0),
        DiscreteAction(dx=0.707, dy=0.707, dtheta=-1.0),
        DiscreteAction(dx=0.707, dy=-0.707, dtheta=1.0),
        DiscreteAction(dx=0.707, dy=-0.707, dtheta=-1.0),
        DiscreteAction(dx=-0.707, dy=0.707, dtheta=1.0),
        DiscreteAction(dx=-0.707, dy=0.707, dtheta=-1.0),
        DiscreteAction(dx=-0.707, dy=-0.707, dtheta=1.0),
        DiscreteAction(dx=-0.707, dy=-0.707, dtheta=-1.0),

        # Rotation
        DiscreteAction(dtheta=1.0),
        DiscreteAction(dtheta=-1.0),

        # Move + rotate
        DiscreteAction(dx=1.0, dtheta=1),
        DiscreteAction(dx=1.0, dtheta=-1),
        DiscreteAction(dx=-1.0, dtheta=1),
        DiscreteAction(dx=-1.0, dtheta=-1),

        # Shooting
        DiscreteAction(shoot=1.0),
    ]


def micro_movement_action_set():
    return [
        DiscreteAction(),  # idle

        # Cardinal movement
        DiscreteAction(dx=1.0),
        DiscreteAction(dx=-1.0),
        DiscreteAction(dy=1.0),
        DiscreteAction(dy=-1.0),

        # Micro movement
        DiscreteAction(dx=0.5),
        DiscreteAction(dx=-0.5),
        DiscreteAction(dy=0.5),
        DiscreteAction(dy=-0.5),

        # Diagonal (normalized)
        DiscreteAction(dx=0.707, dy=0.707),
        DiscreteAction(dx=0.707, dy=-0.707),
        DiscreteAction(dx=-0.707, dy=0.707),
        DiscreteAction(dx=-0.707, dy=-0.707),

        # Rotation
        DiscreteAction(dtheta=1.0),
        DiscreteAction(dtheta=-1.0),

        # Move + rotate
        DiscreteAction(dx=1.0, dtheta=0.3),
        DiscreteAction(dx=1.0, dtheta=-0.3),
        DiscreteAction(dx=-1.0, dtheta=0.3),
        DiscreteAction(dx=-1.0, dtheta=-0.3),

        # Shooting
        DiscreteAction(dx=0.8, shoot=1.0),
        DiscreteAction(dtheta=0.5, shoot=1.0),
    ]
