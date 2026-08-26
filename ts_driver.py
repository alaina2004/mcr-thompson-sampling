import itertools
import random
import numpy as np
import pickle

from ts_evaluator import evaluate_arm  # your GA + docking evaluator


class ArmStat:
    def __init__(self, mu=0.0, tau2=1.0, sigma2=1.0):
        self.mu = mu
        self.tau2 = tau2
        self.sigma2 = sigma2
        self.count = 0

    def sample(self):
        return np.random.normal(self.mu, np.sqrt(self.tau2))

    def update(self, observed_reward):
        tau2 = max(self.tau2, 1e-8)
        sigma2 = max(self.sigma2, 1e-8)
        denom = (1.0 / tau2) + (1.0 / sigma2)
        posterior_variance = 1.0 / denom
        posterior_mean = posterior_variance * (self.mu / tau2 + observed_reward / sigma2)
        self.mu = posterior_mean
        self.tau2 = max(posterior_variance, 1e-8)
        self.count += 1

    def get_confidence_interval(self, z=1.96):
        half_width = z * np.sqrt(self.tau2)
        return self.mu - half_width, self.mu + half_width


class ThompsonSamplingCombinatorial:
    def __init__(
        self,
        r1_list,
        r2_list,
        r3_list,
        reaction,
        core,
        receptor_pdbqt,
        ga_kwargs,
        prior_mean=0.0,
        prior_variance=1.0,
        observation_std=1.0,
        warmup_k=3,
        max_arms=None,
        seed=None,
        epsilon=0.0,
        do_warmup=True,
        lower_is_better=True,  # treat lower raw score as better
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.reaction = reaction
        self.core = core
        self.receptor_pdbqt = receptor_pdbqt
        self.ga_kwargs = ga_kwargs
        self.epsilon = epsilon
        self.lower_is_better = lower_is_better

        # failure raw score for evaluate_arm
        self.failure_raw = 1e6 if self.lower_is_better else -1e6

        full_arms = list(itertools.product(r1_list, r2_list, r3_list))
        if max_arms is not None and len(full_arms) > max_arms:
            self.arms = random.sample(full_arms, max_arms)
        else:
            self.arms = full_arms

        self.arm_stats = {
            arm: ArmStat(mu=prior_mean, tau2=prior_variance, sigma2=observation_std ** 2)
            for arm in self.arms
        }

        self.history = []  # list of (arm, raw_score, reward, best_smiles, best_mol)
        self.failure_counts = {arm: 0 for arm in self.arms}
        self.warmup_k = min(warmup_k, len(self.arms))
        if do_warmup:
            self._warmup()

    def _score_to_reward(self, raw_score):
        reward = -raw_score if self.lower_is_better else raw_score
        # cap extreme values so one outlier doesn't dominate
        return max(min(reward, 1e3), -1e3)

    def seed_initial(self, initial_list):
        """
        initial_list: list of tuples (arm, raw_score, best_smiles, best_mol)
        """
        for arm, raw_score, best_smiles, best_mol in initial_list:
            if arm not in self.arm_stats:
                continue
            reward = self._score_to_reward(raw_score)
            self.arm_stats[arm].update(reward)
            self.history.append((arm, raw_score, reward, best_smiles, best_mol))
            # track failure if it's a failure raw_score
            if best_mol is None or raw_score == self.failure_raw:
                self.failure_counts[arm] += 1

    def _warmup(self):
        for arm in random.sample(self.arms, self.warmup_k):
            raw_score, best_smiles, best_mol = evaluate_arm(
                arm, self.reaction, self.core, self.receptor_pdbqt, self.ga_kwargs, failure_score=self.failure_raw
            )
            reward = self._score_to_reward(raw_score)
            self.arm_stats[arm].update(reward)
            self.history.append((arm, raw_score, reward, best_smiles, best_mol))
            if best_mol is None or raw_score == self.failure_raw:
                self.failure_counts[arm] += 1

    def select_and_update(self):
        if self.epsilon > 0 and random.random() < self.epsilon:
            chosen = random.choice(self.arms)
        else:
            sampled = {arm: stat.sample() for arm, stat in self.arm_stats.items()}
            chosen = max(sampled, key=sampled.get)

        raw_score, best_smiles, best_mol = evaluate_arm(
            chosen, self.reaction, self.core, self.receptor_pdbqt, self.ga_kwargs, failure_score=self.failure_raw
        )
        reward = self._score_to_reward(raw_score)
        self.arm_stats[chosen].update(reward)
        self.history.append((chosen, raw_score, reward, best_smiles, best_mol))

        if best_mol is None or raw_score == self.failure_raw:
            self.failure_counts[chosen] += 1
        else:
            self.failure_counts[chosen] = 0  # reset on success

        # if an arm has failed repeatedly, suppress further sampling
        if self.failure_counts[chosen] >= 5:
            # collapse its variance so it's unlikely to be picked
            self.arm_stats[chosen].tau2 = 1e-8

        return chosen, raw_score, reward, best_smiles, best_mol

    def run(self, iterations=20):
        for _ in range(iterations):
            self.select_and_update()
        return self.history

    def best_so_far(self):
        if not self.history:
            return None, None, None, None, None, (None, None)
        # choose lowest raw_score (since lower_is_better)
        arm, raw_score, reward, best_smiles, best_mol = min(self.history, key=lambda x: x[1])
        stat = self.arm_stats.get(arm)
        ci = stat.get_confidence_interval() if stat else (None, None)
        return arm, raw_score, reward, best_smiles, best_mol, ci

    def save_state(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "arm_stats": self.arm_stats,
                "history": self.history,
                "arms": self.arms,
                "failure_counts": self.failure_counts,
            }, f)

    def load_state(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.arm_stats = data["arm_stats"]
        self.history = data["history"]
        self.arms = data["arms"]
        self.failure_counts = data.get("failure_counts", {arm: 0 for arm in self.arms})
