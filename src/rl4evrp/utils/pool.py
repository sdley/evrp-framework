from typing import Callable, Dict, Sequence, Union


InstanceProvider = Union[Sequence[Dict], Callable[[int], Dict]]


class OnTheFlyInstancePool:
    """
    Deterministic, memory-efficient instance pool.

    Instances are generated on access from a seed schedule rather than being
    pre-materialized in RAM — behaves like a fixed-length sequence but never
    stores more than one instance at a time.
    """

    def __init__(self, generate_fn: Callable[[int], Dict], size: int, seed_offset: int = 0):
        if size <= 0:
            raise ValueError("size must be > 0")
        self.generate_fn = generate_fn
        self.size = int(size)
        self.seed_offset = int(seed_offset)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict:
        return self.generate_fn(seed=self.seed_offset + int(idx) % self.size)


def _resolve_training_instance(train_instances: InstanceProvider, episode: int) -> Dict:
    """Resolve one training instance from either a sequence or callable provider."""
    if callable(train_instances):
        inst = train_instances(episode)
    else:
        if len(train_instances) == 0:
            raise ValueError("train_instances must not be empty")
        inst = train_instances[episode % len(train_instances)]

    if not isinstance(inst, dict):
        raise TypeError("Each training instance must be a dict")

    return dict(inst)
