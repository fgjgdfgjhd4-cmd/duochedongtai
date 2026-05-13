import math

import torch

from gpu_config import DTYPE


def fitness_from_cost(cost):
    return torch.where(cost >= 0, 1.0 / (1.0 + cost), 1.0 + torch.abs(cost))


class GpuPSO:
    def __init__(self, env, population=256, iterations=30, speed_bounds=(5.0, 20.0), turn_bounds=(-math.pi / 4, math.pi / 4)):
        self.env = env
        self.population = population
        self.iterations = iterations
        self.low = torch.tensor([speed_bounds[0], turn_bounds[0]], device=env.device, dtype=DTYPE)
        self.high = torch.tensor([speed_bounds[1], turn_bounds[1]], device=env.device, dtype=DTYPE)
        self.v_low = -3.0
        self.v_high = 3.0

    def _random_actions(self, shape):
        return self.low + torch.rand(shape, device=self.env.device, dtype=DTYPE) * (self.high - self.low)

    def optimize(self):
        x = self._random_actions((self.population, self.env.agent_num, 2))
        v = torch.empty_like(x).uniform_(self.v_low, self.v_high)
        fit = fitness_from_cost(self.env.evaluate_batch(x))
        p_best = x.clone()
        p_fit = fit.clone()
        g_idx = torch.argmax(fit)
        g_best = x[g_idx].clone()
        g_fit = fit[g_idx].clone()

        for _ in range(self.iterations):
            r1 = torch.rand_like(x)
            r2 = torch.rand_like(x)
            v = (0.8 * v + 2.0 * r1 * (p_best - x) + 2.0 * r2 * (g_best[None, :, :] - x)).clamp(self.v_low, self.v_high)
            x = (x + v).clamp(self.low, self.high)
            fit = fitness_from_cost(self.env.evaluate_batch(x))
            improved = fit > p_fit
            p_best = torch.where(improved[:, None, None], x, p_best)
            p_fit = torch.where(improved, fit, p_fit)
            candidate_idx = torch.argmax(p_fit)
            if p_fit[candidate_idx] > g_fit:
                g_best = p_best[candidate_idx].clone()
                g_fit = p_fit[candidate_idx].clone()

        return self.repair(g_best)

    def repair(self, actions):
        cost = self.env.evaluate_batch(actions.unsqueeze(0))[0]
        if torch.isfinite(cost):
            return actions
        repaired = actions.clone()
        repaired[:, 0] = repaired[:, 0].clamp(self.low[0], self.high[0])
        repaired[:, 1] = repaired[:, 1].clamp(self.low[1], self.high[1])
        return repaired


class GpuCPSO(GpuPSO):
    def optimize(self):
        x = self._random_actions((self.population, self.env.agent_num, 2))
        v = torch.empty_like(x).uniform_(self.v_low, self.v_high)
        fit = fitness_from_cost(self.env.evaluate_batch(x))
        p_best = x.clone()
        p_fit = fit.clone()
        g_idx = torch.argmax(fit)
        g_best = x[g_idx].clone()
        g_fit = fit[g_idx].clone()
        w = 0.8
        temp = 0.025

        for _ in range(self.iterations):
            r1 = torch.rand_like(x)
            r2 = torch.rand_like(x)
            v = (w * v + 1.5 * r1 * (p_best - x) + 1.5 * r2 * (g_best[None, :, :] - x)).clamp(self.v_low, self.v_high)
            x = (x + v).clamp(self.low, self.high)
            fit = fitness_from_cost(self.env.evaluate_batch(x))
            improved = fit > p_fit
            accept_worse = torch.rand_like(fit) < torch.exp(((fit - p_fit) / (p_fit + 1e-6)).clamp(max=0.0) / temp)
            accept = improved | accept_worse
            p_best = torch.where(accept[:, None, None], x, p_best)
            p_fit = torch.where(accept, fit, p_fit)
            candidate_idx = torch.argmax(p_fit)
            if p_fit[candidate_idx] > g_fit:
                g_best = p_best[candidate_idx].clone()
                g_fit = p_fit[candidate_idx].clone()
            w *= 0.99
            temp *= 0.99

        return self.repair(g_best)


class GpuABC:
    """Vectorized ABC-style optimizer.

    This keeps the original project's high-level idea, but evaluates all food
    sources in batches and avoids per-source Python objective calls.
    """

    def __init__(self, env, population=256, iterations=30, speed_bounds=(5.0, 20.0), turn_bounds=(-math.pi / 4, math.pi / 4)):
        self.env = env
        self.population = population
        self.iterations = iterations
        self.low = torch.tensor([speed_bounds[0], turn_bounds[0]], device=env.device, dtype=DTYPE)
        self.high = torch.tensor([speed_bounds[1], turn_bounds[1]], device=env.device, dtype=DTYPE)

    def _random_actions(self):
        return self.low + torch.rand((self.population, self.env.agent_num, 2), device=self.env.device, dtype=DTYPE) * (self.high - self.low)

    def optimize(self):
        sources = self._random_actions()
        fit = fitness_from_cost(self.env.evaluate_batch(sources))
        limit = max(5, self.iterations // 2)
        trials = torch.zeros(self.population, device=self.env.device, dtype=torch.int64)

        for _ in range(self.iterations):
            partner_idx = torch.randint(0, self.population, (self.population,), device=self.env.device)
            phi = torch.empty_like(sources).uniform_(-1.0, 1.0)
            candidates = (sources + phi * (sources - sources[partner_idx])).clamp(self.low, self.high)
            cand_fit = fitness_from_cost(self.env.evaluate_batch(candidates))
            improved = cand_fit > fit
            sources = torch.where(improved[:, None, None], candidates, sources)
            fit = torch.where(improved, cand_fit, fit)
            trials = torch.where(improved, torch.zeros_like(trials), trials + 1)

            probs = fit / (fit.sum() + 1e-8)
            selected = torch.multinomial(probs, self.population, replacement=True)
            partner_idx = torch.randint(0, self.population, (self.population,), device=self.env.device)
            phi = torch.empty_like(sources).uniform_(-1.0, 1.0)
            candidates = (sources[selected] + phi * (sources[selected] - sources[partner_idx])).clamp(self.low, self.high)
            cand_fit = fitness_from_cost(self.env.evaluate_batch(candidates))
            selected_fit = fit[selected]
            improved = cand_fit > selected_fit
            improved_idx = selected[improved]
            if improved_idx.numel() > 0:
                sources[improved_idx] = candidates[improved]
                fit[improved_idx] = cand_fit[improved]
                trials[improved_idx] = 0

            stale = trials > limit
            if torch.any(stale):
                sources[stale] = self._random_actions()[: int(stale.sum().item())]
                fit[stale] = fitness_from_cost(self.env.evaluate_batch(sources[stale]))
                trials[stale] = 0

        return sources[torch.argmax(fit)].clone()


class GpuOriginABC(GpuABC):
    """GPU port of Compare_Method_ABC.swarm.ABC_origin."""

    def optimize(self):
        sources = self._random_actions()
        fit = fitness_from_cost(self.env.evaluate_batch(sources))
        trials = torch.zeros(self.population, device=self.env.device, dtype=torch.int64)
        limit = max(5, self.iterations // 2)

        for _ in range(self.iterations):
            partner = torch.randint(0, self.population, (self.population,), device=self.env.device)
            dim_agent = torch.randint(0, self.env.agent_num, (self.population,), device=self.env.device)
            dim_action = torch.randint(0, 2, (self.population,), device=self.env.device)
            r = torch.empty((self.population,), device=self.env.device, dtype=DTYPE).uniform_(-1.0, 1.0)
            candidates = sources.clone()
            row = torch.arange(self.population, device=self.env.device)
            candidates[row, dim_agent, dim_action] = (
                sources[row, dim_agent, dim_action]
                + r * (sources[row, dim_agent, dim_action] - sources[partner, dim_agent, dim_action])
            )
            candidates = candidates.clamp(self.low, self.high)
            cand_fit = fitness_from_cost(self.env.evaluate_batch(candidates))
            improved = cand_fit > fit
            sources = torch.where(improved[:, None, None], candidates, sources)
            fit = torch.where(improved, cand_fit, fit)
            trials = torch.where(improved, torch.zeros_like(trials), trials + 1)

            probs = fit / (fit.sum() + 1e-8)
            selected = torch.multinomial(probs, self.population, replacement=True)
            partner = torch.randint(0, self.population, (self.population,), device=self.env.device)
            dim_agent = torch.randint(0, self.env.agent_num, (self.population,), device=self.env.device)
            dim_action = torch.randint(0, 2, (self.population,), device=self.env.device)
            r = torch.empty((self.population,), device=self.env.device, dtype=DTYPE).uniform_(-1.0, 1.0)
            candidates = sources[selected].clone()
            row = torch.arange(self.population, device=self.env.device)
            candidates[row, dim_agent, dim_action] = (
                sources[selected, dim_agent, dim_action]
                + r * (sources[selected, dim_agent, dim_action] - sources[partner, dim_agent, dim_action])
            )
            candidates = candidates.clamp(self.low, self.high)
            cand_fit = fitness_from_cost(self.env.evaluate_batch(candidates))
            improved = cand_fit > fit[selected]
            improved_idx = selected[improved]
            if improved_idx.numel() > 0:
                sources[improved_idx] = candidates[improved]
                fit[improved_idx] = cand_fit[improved]
                trials[improved_idx] = 0

            stale = trials > limit
            if torch.any(stale):
                sources[stale] = self._random_actions()[: int(stale.sum().item())]
                fit[stale] = fitness_from_cost(self.env.evaluate_batch(sources[stale]))
                trials[stale] = 0

        return sources[torch.argmax(fit)].clone()


class GpuIABC(GpuABC):
    """GPU port of abc_for_map.ABC with configurable strategy probabilities."""

    def __init__(self, *args, probabilities=None, **kwargs):
        super().__init__(*args, **kwargs)
        if probabilities is None:
            probabilities = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        self.set_probability(probabilities)

    def set_probability(self, probabilities):
        probs = torch.as_tensor(probabilities, dtype=DTYPE, device=self.env.device)
        self.probabilities = probs / (probs.sum() + 1e-8)

    def optimize(self):
        sources = self._random_actions()
        fit = fitness_from_cost(self.env.evaluate_batch(sources))
        trials = torch.zeros(self.population, device=self.env.device, dtype=torch.int64)
        limit = max(5, self.iterations // 2)

        for iter_idx in range(self.iterations):
            idx1 = torch.randint(0, self.population, (self.population,), device=self.env.device)
            idx2 = torch.randint(0, self.population, (self.population,), device=self.env.device)
            phi1 = torch.empty((self.population, 1, 1), device=self.env.device, dtype=DTYPE).uniform_(-1.0, 1.0)
            phi2 = torch.empty((self.population, 1, 1), device=self.env.device, dtype=DTYPE).uniform_(-1.0, 1.0)
            candidates = (sources + phi1 * (sources[idx1] - sources) + phi2 * (sources[idx2] - sources)).clamp(self.low, self.high)
            cand_fit = fitness_from_cost(self.env.evaluate_batch(candidates))
            improved = cand_fit > fit
            sources = torch.where(improved[:, None, None], candidates, sources)
            fit = torch.where(improved, cand_fit, fit)
            trials = torch.where(improved, torch.zeros_like(trials), trials + 1)

            selected = torch.multinomial(fit / (fit.sum() + 1e-8), self.population, replacement=True)
            candidates = sources[selected].clone()
            strategy = torch.multinomial(self.probabilities, self.population, replacement=True)
            best_idx = torch.argmax(fit)
            global_best = sources[best_idx]
            row = torch.arange(self.population, device=self.env.device)
            dim_agent = torch.randint(0, self.env.agent_num, (self.population,), device=self.env.device)
            dim_action = torch.randint(0, 2, (self.population,), device=self.env.device)

            spiral = strategy == 0
            if torch.any(spiral):
                r = torch.empty((int(spiral.sum().item()),), device=self.env.device, dtype=DTYPE).uniform_(-1.0, 1.0)
                b = 1.0 - float(iter_idx + 1) / float(max(1, self.iterations))
                rr = row[spiral]
                aa = dim_agent[spiral]
                dd = dim_action[spiral]
                candidates[rr, aa, dd] = (
                    global_best[aa, dd]
                    + torch.abs(global_best[aa, dd] - candidates[rr, aa, dd])
                    * torch.exp(torch.as_tensor(b, device=self.env.device, dtype=DTYPE) * r)
                    * torch.cos(2.0 * math.pi * r)
                )

            guided = strategy == 1
            if torch.any(guided):
                partner = torch.randint(0, self.population, (int(guided.sum().item()),), device=self.env.device)
                rr = row[guided]
                aa = dim_agent[guided]
                dd = dim_action[guided]
                phi1 = torch.empty_like(rr, dtype=DTYPE).uniform_(-1.0, 1.0)
                phi2 = torch.empty_like(rr, dtype=DTYPE).uniform_(-1.0, 1.0)
                c2 = 2.0 * trials[selected[guided]].to(DTYPE) / float(limit)
                c3 = 2.0 * (1.0 - trials[selected[guided]].to(DTYPE) / float(limit))
                candidates[rr, aa, dd] = (
                    sources[selected[guided], aa, dd]
                    + 0.5 * phi1 * (sources[partner, aa, dd] - sources[selected[guided], aa, dd])
                    + c2 * phi2 * (global_best[aa, dd] - sources[selected[guided], aa, dd])
                    + c3 * (global_best[aa, dd] - sources[selected[guided], aa, dd])
                )

            diff = strategy == 2
            if torch.any(diff):
                idx1 = torch.randint(0, self.population, (int(diff.sum().item()),), device=self.env.device)
                idx2 = torch.randint(0, self.population, (int(diff.sum().item()),), device=self.env.device)
                rr = row[diff]
                aa = dim_agent[diff]
                dd = dim_action[diff]
                factor = torch.empty_like(rr, dtype=DTYPE).uniform_(-1.0, 1.0)
                candidates[rr, aa, dd] = sources[selected[diff], aa, dd] + factor * (sources[idx1, aa, dd] - sources[idx2, aa, dd])

            candidates = candidates.clamp(self.low, self.high)
            cand_fit = fitness_from_cost(self.env.evaluate_batch(candidates))
            improved = cand_fit > fit[selected]
            improved_idx = selected[improved]
            if improved_idx.numel() > 0:
                sources[improved_idx] = candidates[improved]
                fit[improved_idx] = cand_fit[improved]
                trials[improved_idx] = 0

            stale = trials > limit
            if torch.any(stale):
                sources[stale] = self._random_actions()[: int(stale.sum().item())]
                fit[stale] = fitness_from_cost(self.env.evaluate_batch(sources[stale]))
                trials[stale] = 0

        return sources[torch.argmax(fit)].clone()


class GpuABCL(GpuABC):
    """GPU port of Compare_Method_ABCL.ABCL_MAP.ABCL."""

    def optimize(self):
        sources = self._random_actions()
        fit = fitness_from_cost(self.env.evaluate_batch(sources))
        trials = torch.zeros(self.population, device=self.env.device, dtype=torch.int64)
        limit = max(5, self.iterations // 2)

        for _ in range(self.iterations):
            best_idx = torch.argmax(fit)
            global_best = sources[best_idx]
            idx1 = torch.randint(0, self.population, (self.population,), device=self.env.device)
            idx2 = torch.randint(0, self.population, (self.population,), device=self.env.device)
            row = torch.arange(self.population, device=self.env.device)
            dim_agent = torch.randint(0, self.env.agent_num, (self.population,), device=self.env.device)
            dim_action = torch.randint(0, 2, (self.population,), device=self.env.device)
            phi1 = torch.empty((self.population,), device=self.env.device, dtype=DTYPE).uniform_(-1.0, 1.0)
            phi2 = torch.empty((self.population,), device=self.env.device, dtype=DTYPE).uniform_(-1.0, 1.0)
            candidates = sources.clone()
            candidates[row, dim_agent, dim_action] = (
                sources[row, dim_agent, dim_action]
                + phi1 * (global_best[dim_agent, dim_action] - sources[row, dim_agent, dim_action])
                + phi2 * (sources[idx1, dim_agent, dim_action] - sources[idx2, dim_agent, dim_action])
            )
            candidates = candidates.clamp(self.low, self.high)
            cand_fit = fitness_from_cost(self.env.evaluate_batch(candidates))
            improved = cand_fit > fit
            sources = torch.where(improved[:, None, None], candidates, sources)
            fit = torch.where(improved, cand_fit, fit)
            trials = torch.where(improved, torch.zeros_like(trials), trials + 1)

            partner = torch.randint(0, self.population, (self.population,), device=self.env.device)
            dim_agent = torch.randint(0, self.env.agent_num, (self.population,), device=self.env.device)
            dim_action = torch.randint(0, 2, (self.population,), device=self.env.device)
            r = torch.rand((self.population,), device=self.env.device, dtype=DTYPE)
            candidates = sources.clone()
            direction = torch.where(
                (fit > fit[partner])[:, None],
                sources[row, dim_agent, dim_action][:, None] - sources[partner, dim_agent, dim_action][:, None],
                sources[partner, dim_agent, dim_action][:, None] - sources[row, dim_agent, dim_action][:, None],
            ).squeeze(1)
            candidates[row, dim_agent, dim_action] = sources[row, dim_agent, dim_action] + r * direction
            candidates = candidates.clamp(self.low, self.high)
            cand_fit = fitness_from_cost(self.env.evaluate_batch(candidates))
            improved = cand_fit > fit
            sources = torch.where(improved[:, None, None], candidates, sources)
            fit = torch.where(improved, cand_fit, fit)
            trials = torch.where(improved, torch.zeros_like(trials), trials + 1)

            stale = trials > limit
            if torch.any(stale):
                best_idx = torch.argmax(fit)
                random_part = torch.empty((int(stale.sum().item()), 1, 1), device=self.env.device, dtype=DTYPE).uniform_(-1.0, 1.0)
                scouts = sources[best_idx].unsqueeze(0).repeat(int(stale.sum().item()), 1, 1)
                scout_agent = torch.randint(0, self.env.agent_num, (int(stale.sum().item()),), device=self.env.device)
                scout_action = torch.randint(0, 2, (int(stale.sum().item()),), device=self.env.device)
                scout_row = torch.arange(int(stale.sum().item()), device=self.env.device)
                scouts[scout_row, scout_agent, scout_action] = (
                    sources[best_idx, scout_agent, scout_action]
                    + random_part.squeeze(-1).squeeze(-1) * (self.high[scout_action] - self.low[scout_action])
                )
                sources[stale] = scouts.clamp(self.low, self.high)
                fit[stale] = fitness_from_cost(self.env.evaluate_batch(sources[stale]))
                trials[stale] = 0

        return sources[torch.argmax(fit)].clone()
