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

        # Movement: Full speed
        DiscreteAction(dx=1.0),
        DiscreteAction(dx=-1.0),
        DiscreteAction(dy=1.0),
        DiscreteAction(dy=-1.0),

        # Movement: Precision/Micro-adjustments
        DiscreteAction(dx=0.5),
        DiscreteAction(dx=-0.5),
        DiscreteAction(dy=0.5),
        DiscreteAction(dy=-0.5),

        # Diagonal (Normalized)
        DiscreteAction(dx=0.75, dy=0.75),
        DiscreteAction(dx=0.75, dy=-0.75),
        # DiscreteAction(dx=-0.75, dy=0.75),
        # DiscreteAction(dx=-0.75, dy=-0.75),

        # Rotation: Fast and Slow
        DiscreteAction(dtheta=1.0),  # Quick pivot
        DiscreteAction(dtheta=-1.0),
        DiscreteAction(dtheta=0.3),  # Fine-tuning aim
        DiscreteAction(dtheta=-0.3),

        # Offensive: Move & Shoot
        DiscreteAction(shoot=1.0),
        DiscreteAction(dx=1.0, shoot=1.0),  # Charging shot
        DiscreteAction(dtheta=0.5, shoot=1.0),  # Slap shot (rotating into puck)

        # Defensive: Retreating
        DiscreteAction(dx=-1.0, dy=0.5),  # Diagonal retreat
        DiscreteAction(dx=-1.0, dy=-0.5),
    ]
