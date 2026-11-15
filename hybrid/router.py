"""Router: freeze confident Stage 1 decisions, send the rest to the Stage 2 arbiter."""
from dataclasses import dataclass, field

from .data import TARGETS
from .triggers import BLOCKING, complex_repair, detect


@dataclass
class RouterConfig:
    t_low: float = 0.15
    t_high: float = 0.85
    t_mid: float = 0.5
    use_triggers: bool = True
    check_101: bool = True


@dataclass
class Route:
    frozen: dict
    contested: list
    triggers: dict = field(default_factory=dict)
    stage1: list = field(default_factory=list)

    @property
    def escalate(self):
        return bool(self.contested)


def route(probs, mc_id, text, cfg: RouterConfig):
    """probs are p_102..p_106. Returns which categories are frozen (0/1) and which are contested."""
    fired = detect(text) if cfg.use_triggers else {}
    risky = set().union(*(v for k, v in fired.items() if k in BLOCKING)) if fired else set()
    frozen, contested = {}, []
    for c, p in zip(TARGETS, probs):
        if c == mc_id:
            continue
        if p < cfg.t_low:
            frozen[c] = 0
        elif p > cfg.t_high and c not in risky:
            frozen[c] = 1
        else:
            contested.append(c)
    if mc_id != 101:
        if cfg.check_101 and complex_repair(text):
            contested.insert(0, 101)
        else:
            frozen[101] = 0
    stage1 = sorted(c for c, p in zip(TARGETS, probs) if p >= cfg.t_mid and c != mc_id)
    return Route(frozen=frozen, contested=contested, triggers={k: sorted(v) for k, v in fired.items()}, stage1=stage1)


def merge(r: Route, llm_labels):
    """Final labels: frozen decisions are kept, the arbiter may only decide contested categories."""
    llm = set(llm_labels) & set(r.contested)
    return sorted({c for c, v in r.frozen.items() if v == 1} | llm)
