"""Independent inference-only rollout for the replayed public Kinetix actor."""

from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jnp
from kinetix.environment.env import make_kinetix_env
from kinetix.environment.spaces import ActionType, ObservationType
from kinetix.util.saving import BASE_DIR, load_from_json_file
import numpy as np

from replay_kinetix_weight_edit import LAYERS, read_pinned


LEVEL = "l/h2_car_ramp.json"
LEVEL_SHA256 = "6616869302884dcfa74ec2e8da37b523c13fdc51f4b829eb3ecd7ba5e0ada6b1"
PACKAGE_VERSIONS = {"kinetix-env": "3.0.2", "jax": "0.9.0", "jaxlib": "0.9.0", "jax2d": "1.0.1"}
ACTION_BOUNDS = (0, 3, 6, 9, 12, 14, 16)
BATCH = 8
MAX_STEPS = 256


def evaluate(actors, episodes, seed):
    if episodes not in range(BATCH, 257, BATCH):
        raise ValueError("unsupported number of episodes")
    for package, expected in PACKAGE_VERSIONS.items():
        if version(package) != expected:
            raise ValueError(f"evaluation requires {package}=={expected}")
    level_path = Path(BASE_DIR) / LEVEL
    read_pinned(level_path, LEVEL_SHA256)
    state, static_params, params = load_from_json_file(str(level_path))
    params = params.replace(dense_reward_scale=0.2, max_timesteps=MAX_STEPS)
    env = make_kinetix_env(ActionType.MULTI_DISCRETE, ObservationType.SYMBOLIC_FLAT, None,
                           params, static_params, auto_reset=False)

    def policy(obs, tensors):
        obs = jnp.clip(jnp.nan_to_num(obs, nan=0, posinf=0, neginf=0), -10, 10)
        for i, (weights, bias) in enumerate(tensors):
            obs = obs @ weights + bias
            if i == 0:
                obs = jax.nn.relu(obs)
            elif i < len(tensors) - 1:
                obs = jnp.tanh(obs)
        return obs

    def rollout(tensors, key, initial_state):
        initial_obs, initial_state = env.reset(key, params, override_reset_state=initial_state)
        init = (initial_obs, initial_state, key, 0, False, False, False, 0.0, -1)

        def active(carry):
            return (carry[3] < MAX_STEPS) & ~carry[4]

        def step(carry):
            obs, sim, rng, count, _, _, _, total_reward, first_motor = carry
            rng, action_rng, physics_rng = jax.random.split(rng, 3)
            logits = policy(obs, tensors)
            action = jnp.stack([
                jax.random.categorical(k, logits[start:end])
                for k, start, end in zip(jax.random.split(action_rng, 6),
                                         ACTION_BOUNDS[:-1], ACTION_BOUNDS[1:], strict=True)
            ]).astype(jnp.int32)
            obs, sim, reward, done, info = env.step(physics_rng, sim, action, params)
            return (obs, sim, rng, count + 1, done, info["GoalR"], reward < -0.5,
                    total_reward + reward, jnp.where(count == 0, action[0], first_motor))

        _, _, _, length, _, goal, red, total, first_motor = jax.lax.while_loop(active, step, init)
        outcome = jnp.where(goal, 0, jnp.where(red, 1, jnp.where(length == MAX_STEPS, 2, 3)))
        return outcome, length, total, first_motor

    with jax.default_device(jax.devices("cpu")[0]):
        obs, _ = env.reset(jax.random.key(seed), params, override_reset_state=state)
        if obs.shape != (679,) or static_params.frame_skip != 2:
            raise ValueError("unexpected Kinetix observation layout or frame skip")
        # Keep compilation and memory bounded regardless of total trial count.
        run_batch = jax.jit(jax.vmap(rollout, in_axes=(None, 0, None)))
        keys = jax.random.split(jax.random.key(seed), episodes)
        results, episode_rows = {}, {}
        for name, actor in actors.items():
            tensors = tuple((jnp.asarray(actor[f"{layer}.W"], dtype=jnp.float32),
                             jnp.asarray(actor[f"{layer}.b"], dtype=jnp.float32)) for layer in LAYERS)
            outcome, length, reward, first_motor = np.concatenate([
                np.asarray(run_batch(tensors, keys[i:i + BATCH], state))
                for i in range(0, episodes, BATCH)
            ], axis=1)
            episode_rows[name] = np.stack([outcome, length, reward, first_motor])
            results[name] = {
                **{key: int(np.count_nonzero(outcome == i))
                   for i, key in enumerate(("successes", "red", "timeouts", "invalid"))},
                "mean_steps": float(length.mean()),
                "mean_return": float(reward.mean()),
                "first_motor_counts_off_forward_reverse": [
                    int(np.count_nonzero(first_motor == i)) for i in range(3)
                ],
            }
    return {
        "level": LEVEL, "level_sha256": LEVEL_SHA256, "episodes_per_actor": episodes,
        "seed": seed, "device": "cpu", "package_versions": PACKAGE_VERSIONS,
        "sampling": "independent categorical action entries, same episode keys for all actors",
        "restored_matches_baseline_per_episode": bool(np.array_equal(episode_rows["base"], episode_rows["restored"])),
        "results": results,
    }
