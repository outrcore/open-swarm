"""OutrCore Open Swarm v0.1 — Token, prefix-cache, and spend-cap metering."""
from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any


class SpendCapExceeded(RuntimeError):
    """Raised when a swarm run crosses its configured hard USD spend cap."""


@dataclass(frozen=True)
class PricingSchedule:
    """USD per 1M tokens for uncached input, cached input reads, and output."""

    input_usd_per_m: float = 0.055
    cached_input_usd_per_m: float = 0.0125
    output_usd_per_m: float = 0.50

    def cost_usd(self, uncached_in: int, cached_in: int, output_tokens: int) -> float:
        return (
            (max(0, uncached_in) * self.input_usd_per_m)
            + (max(0, cached_in) * self.cached_input_usd_per_m)
            + (max(0, output_tokens) * self.output_usd_per_m)
        ) / 1_000_000.0


@dataclass
class AgentMetrics:
    agent_id: str
    calls: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    total_gen_s: float = 0.0
    ttft_ms_samples: list[float] = field(default_factory=list)
    wall_s: float = 0.0

    @property
    def uncached_tokens(self) -> int:
        return max(0, self.prompt_tokens - self.cached_tokens)

    @property
    def cache_hit_pct(self) -> float:
        return (self.cached_tokens / self.prompt_tokens * 100.0) if self.prompt_tokens > 0 else 0.0

    @property
    def avg_tok_s(self) -> float:
        return (self.completion_tokens / self.total_gen_s) if self.total_gen_s > 0 else 0.0


class RunMeter:
    """Thread-safe token, prefix-cache, and hard USD spend-cap tracker for Open Swarm runs."""

    def __init__(self, pricing: PricingSchedule | None = None, spend_cap_usd: float | None = None) -> None:
        self.pricing = pricing or PricingSchedule()
        self.spend_cap_usd = spend_cap_usd if (spend_cap_usd is None or spend_cap_usd > 0) else 0.0
        self._lock = threading.Lock()
        self.agents: dict[str, AgentMetrics] = {}

    def _agent(self, agent_id: str) -> AgentMetrics:
        if agent_id not in self.agents:
            self.agents[agent_id] = AgentMetrics(agent_id=agent_id)
        return self.agents[agent_id]

    def total_cost_usd_unlocked(self) -> float:
        p_tot = sum(a.prompt_tokens for a in self.agents.values())
        c_tot = sum(a.cached_tokens for a in self.agents.values())
        o_tot = sum(a.completion_tokens for a in self.agents.values())
        return self.pricing.cost_usd(max(0, p_tot - c_tot), c_tot, o_tot)

    def check_cap(self) -> None:
        if self.spend_cap_usd is None:
            return
        with self._lock:
            cur = self.total_cost_usd_unlocked()
            if cur >= self.spend_cap_usd:
                raise SpendCapExceeded(f"Run cost ${cur:.6f} reached spend cap ${self.spend_cap_usd:.6f}")

    def record_usage(
        self,
        agent_id: str,
        *,
        prompt_tokens: int,
        cached_tokens: int,
        completion_tokens: int,
        gen_s: float,
        ttft_ms: float | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            ag = self._agent(agent_id)
            p_toks = max(0, int(prompt_tokens))
            c_toks = max(0, min(p_toks, int(cached_tokens)))
            o_toks = max(0, int(completion_tokens))
            ag.calls += 1
            ag.prompt_tokens += p_toks
            ag.cached_tokens += c_toks
            ag.completion_tokens += o_toks
            ag.total_gen_s += max(0.0, float(gen_s))
            if ttft_ms is not None and ttft_ms >= 0:
                ag.ttft_ms_samples.append(float(ttft_ms))
            tot_cost = self.total_cost_usd_unlocked()
            if self.spend_cap_usd is not None and tot_cost > self.spend_cap_usd:
                raise SpendCapExceeded(f"Run cost ${tot_cost:.6f} exceeded spend cap ${self.spend_cap_usd:.6f}")
            return {
                "agent_id": agent_id,
                "calls": ag.calls,
                "run_cost_usd": round(tot_cost, 6),
            }

    def set_agent_wall(self, agent_id: str, wall_s: float) -> None:
        with self._lock:
            self._agent(agent_id).wall_s = max(0.0, float(wall_s))

    def summary(self, wall_s: float) -> dict[str, Any]:
        with self._lock:
            p_tot = sum(a.prompt_tokens for a in self.agents.values())
            c_tot = sum(a.cached_tokens for a in self.agents.values())
            o_tot = sum(a.completion_tokens for a in self.agents.values())
            u_tot = max(0, p_tot - c_tot)
            cost = self.pricing.cost_usd(u_tot, c_tot, o_tot)
            all_ttft = [x for a in self.agents.values() for x in a.ttft_ms_samples]
            return {
                "wall_s": round(float(wall_s), 3),
                "calls": sum(a.calls for a in self.agents.values()),
                "prompt_tokens": p_tot,
                "uncached_input_tokens": u_tot,
                "cached_input_tokens": c_tot,
                "completion_tokens": o_tot,
                "cache_hit_pct": round((c_tot / p_tot * 100.0) if p_tot > 0 else 0.0, 2),
                "aggregate_tok_s": round((o_tot / wall_s) if wall_s > 0 else 0.0, 2),
                "mean_ttft_ms": round(sum(all_ttft) / len(all_ttft), 2) if all_ttft else None,
                "cost_usd": round(cost, 6),
                "spend_cap_usd": self.spend_cap_usd,
                "pricing": {
                    "input_usd_per_m": self.pricing.input_usd_per_m,
                    "cached_input_usd_per_m": self.pricing.cached_input_usd_per_m,
                    "output_usd_per_m": self.pricing.output_usd_per_m,
                },
                "agents": {
                    aid: {
                        "calls": a.calls,
                        "prompt_tokens": a.prompt_tokens,
                        "uncached_tokens": a.uncached_tokens,
                        "cached_tokens": a.cached_tokens,
                        "cache_hit_pct": round(a.cache_hit_pct, 2),
                        "completion_tokens": a.completion_tokens,
                        "avg_tok_s": round(a.avg_tok_s, 2),
                        "wall_s": round(a.wall_s, 3),
                    }
                    for aid, a in sorted(self.agents.items())
                },
            }
