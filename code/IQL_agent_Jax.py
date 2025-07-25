import os
import copy
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any

import chex
import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
#import hydra
#from omegaconf import OmegaConf
#import gymnax
import flashbax as fbx
#import wandb

from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.wrappers.baselines import (
    SMAXLogWrapper,
    MPELogWrapper,
    LogWrapper,
    CTRolloutManager,
)

from jaxmarl.viz.visualizer import Visualizer, SMAXVisualizer
from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX

class ScannedRNN(nn.Module):

    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        hidden_size = ins.shape[-1]
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(hidden_size, *ins.shape[:-1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )


class RNNQNetwork(nn.Module):
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
    action_dim: int
    hidden_dim: int
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, hidden, obs, dones):
        embedding = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        q_vals = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(embedding)

        return hidden, q_vals


@chex.dataclass(frozen=True)
class Timestep:
    obs: dict
    actions: dict
    rewards: dict
    dones: dict
    avail_actions: dict


class CustomTrainState(TrainState):
    target_network_params: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def make_train(config, env):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    eps_scheduler = optax.linear_schedule(
        init_value=config["EPS_START"],
        end_value=config["EPS_FINISH"],
        transition_steps=config["EPS_DECAY"] * config["NUM_UPDATES"],
    )

    def get_greedy_actions(q_vals, valid_actions):
        unavail_actions = 1 - valid_actions
        q_vals = q_vals - (unavail_actions * 1e10)
        return jnp.argmax(q_vals, axis=-1)

    # epsilon-greedy exploration
    def eps_greedy_exploration(rng, q_vals, eps, valid_actions):

        rng_a, rng_e = jax.random.split(
            rng
        )  # a key for sampling random actions and one for picking

        greedy_actions = get_greedy_actions(q_vals, valid_actions)

        # pick random actions from the valid actions
        def get_random_actions(rng, val_action):
            return jax.random.choice(
                rng,
                jnp.arange(val_action.shape[-1]),
                p=val_action * 1.0 / jnp.sum(val_action, axis=-1),
            )

        _rngs = jax.random.split(rng_a, valid_actions.shape[0])
        random_actions = jax.vmap(get_random_actions)(_rngs, valid_actions)

        chosed_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape)
            < eps,  # pick the actions that should be random
            random_actions,
            greedy_actions,
        )
        return chosed_actions

    def batchify(x: dict):
        return jnp.stack([x[agent] for agent in env.agents], axis=0)

    def unbatchify(x: jnp.ndarray):
        return {agent: x[i] for i, agent in enumerate(env.agents)}

    def train(rng):

        # INIT ENV
        original_seed = rng[0]
        rng, _rng = jax.random.split(rng)
        wrapped_env = CTRolloutManager(env, batch_size=config["NUM_ENVS"])
        test_env = CTRolloutManager(
            env, batch_size=config["TEST_NUM_ENVS"]
        )  # batched env for testing (has different batch size)

        # INIT NETWORK AND OPTIMIZER
        network = RNNQNetwork(
            action_dim=wrapped_env.max_action_space,
            hidden_dim=config["HIDDEN_SIZE"],
        )

        def create_agent(rng):
            init_x = (
                jnp.zeros(
                    (1, 1, wrapped_env.obs_size)
                ),  # (time_step, batch_size, obs_size)
                jnp.zeros((1, 1)),  # (time_step, batch size)
            )
            init_hs = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], 1
            )  # (batch_size, hidden_dim)
            network_params = network.init(rng, init_hs, *init_x)

            lr_scheduler = optax.linear_schedule(
                init_value=config["LR"],
                end_value=1e-10,
                transition_steps=(config["NUM_EPOCHS"]) * config["NUM_UPDATES"],
            )

            lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )

            train_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=network_params,
                target_network_params=network_params,
                tx=tx,
            )
            return train_state

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(rng)

        # INIT BUFFER
        # to initalize the buffer is necessary to sample a trajectory to know its strucutre
        def _env_sample_step(env_state, unused):
            rng, key_a, key_s = jax.random.split(
                jax.random.PRNGKey(0), 3
            )  # use a dummy rng here
            key_a = jax.random.split(key_a, env.num_agents)
            actions = {
                agent: wrapped_env.batch_sample(key_a[i], agent)
                for i, agent in enumerate(env.agents)
            }
            avail_actions = wrapped_env.get_valid_actions(env_state.env_state)
            obs, env_state, rewards, dones, infos = wrapped_env.batch_step(
                key_s, env_state, actions
            )
            timestep = Timestep(
                obs=obs,
                actions=actions,
                rewards=rewards,
                dones=dones,
                avail_actions=avail_actions,
            )
            return env_state, timestep

        _, _env_state = wrapped_env.batch_reset(rng)
        _, sample_traj = jax.lax.scan(
            _env_sample_step, _env_state, None, config["NUM_STEPS"]
        )
        sample_traj_unbatched = jax.tree.map(
            lambda x: x[:, 0], sample_traj
        )  # remove the NUM_ENV dim
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=config["BUFFER_SIZE"] // config["NUM_ENVS"],
            min_length_time_axis=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_batch_size=config["NUM_ENVS"],
            sample_sequence_length=1,
            period=1,
        )
        print("init sample_traj shape:", jax.tree.map(lambda x: x.shape, sample_traj))
        buffer_state = buffer.init(sample_traj_unbatched)

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, buffer_state, test_state, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                hs, last_obs, last_dones, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)

                # (num_agents, 1 (dummy time), num_envs, obs_size)
                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]

                #########################################################################################
                print("hstate shape:", hs.shape)  
                print("_obs shape:", _obs.shape)     
                print("_dones shape:", _dones.shape)  
                #########################################################################################

                new_hs, q_vals = jax.vmap(
                    network.apply, in_axes=(None, 0, 0, 0)
                )(  # vmap across the agent dim
                    train_state.params,
                    hs,
                    _obs,
                    _dones,
                )
                q_vals = q_vals.squeeze(
                    axis=1
                )  # (num_agents, num_envs, num_actions) remove the time dim

                # explore
                avail_actions = wrapped_env.get_valid_actions(env_state.env_state)

                eps = eps_scheduler(train_state.n_updates)
                _rngs = jax.random.split(rng_a, env.num_agents)
                actions = jax.vmap(eps_greedy_exploration, in_axes=(0, 0, None, 0))(
                    _rngs, q_vals, eps, batchify(avail_actions)
                )
                actions = unbatchify(actions)
                print("actions", actions)


                new_obs, new_env_state, rewards, dones, infos = wrapped_env.batch_step(
                    rng_s, env_state, actions
                )
                timestep = Timestep(
                    obs=last_obs,
                    actions=actions,
                    rewards=jax.tree.map(lambda x:config.get("REW_SCALE", 1)*x, rewards),
                    dones=last_dones,
                    avail_actions=avail_actions,
                )
                return (new_hs, new_obs, dones, new_env_state, rng), (timestep, infos)

            # step the env (should be a complete rollout)
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = wrapped_env.batch_reset(_rng)
            print("init_obs is a dict with keys:", init_obs.keys())
            for k, v in init_obs.items():
                print(f"init_obs[{k}] shape:", v.shape) 
            init_dones = {
                agent: jnp.zeros((config["NUM_ENVS"]), dtype=bool)
                for agent in env.agents + ["__all__"]
            }
            init_hs = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], len(env.agents), config["NUM_ENVS"]
            )
            expl_state = (init_hs, init_obs, init_dones, env_state)
            rng, _rng = jax.random.split(rng)
            _, (timesteps, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )

            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )  # update timesteps count

            # BUFFER UPDATE
            buffer_traj_batch = jax.tree.map(
                lambda x: jnp.swapaxes(x, 0, 1)[
                    :, np.newaxis
                ],  # put the batch dim first and add a dummy sequence dim
                timesteps,
            )  # (num_envs, 1, time_steps, ...)
            print("Runtime sample_traj shape:", jax.tree.map(lambda x: x.shape, buffer_traj_batch))
            buffer_state = buffer.add(buffer_state, buffer_traj_batch)

            # NETWORKS UPDATE
            def _learn_phase(carry, _):

                train_state, rng = carry
                rng, _rng = jax.random.split(rng)
                minibatch = buffer.sample(buffer_state, _rng).experience
                minibatch = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        x[:, 0], 0, 1
                    ),  # remove the dummy sequence dim (1) and swap batch and temporal dims
                    minibatch,
                )  # (max_time_steps, batch_size, ...)

                # preprocess network input
                init_hs = ScannedRNN.initialize_carry(
                    config["HIDDEN_SIZE"],
                    len(env.agents),
                    config["BUFFER_BATCH_SIZE"],
                )
                # num_agents, timesteps, batch_size, ...
                _obs = batchify(minibatch.obs)
                _dones = batchify(minibatch.dones)
                _actions = batchify(minibatch.actions)
                _rewards = batchify(minibatch.rewards)
                _avail_actions = batchify(minibatch.avail_actions)
                ###################################################################################################
                #print("_obs.shape", _obs.shape)
                #print("_does.shape", _dones.shape)
                #print("init_hs.shape", init_hs.shape)
                ###################################################################################################
                _, q_next_target = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                    train_state.target_network_params,
                    init_hs,
                    _obs,
                    _dones,
                )  # (num_agents, timesteps, batch_size, num_actions)

                def _loss_fn(params):
                    _, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                        params,
                        init_hs,
                        _obs,
                        _dones,
                    )  # (num_agents, timesteps, batch_size, num_actions)

                    # get logits of the chosen actions
                    chosen_action_q_vals = jnp.take_along_axis(
                        q_vals,
                        _actions[..., np.newaxis],
                        axis=-1,
                    ).squeeze(
                        -1
                    )  # (num_agents, timesteps, batch_size,)

                    unavailable_actions = 1 - _avail_actions
                    valid_q_vals = q_vals - (unavailable_actions * 1e10)

                    # get the q values of the next state
                    q_next = jnp.take_along_axis(
                        q_next_target,
                        jnp.argmax(valid_q_vals, axis=-1)[..., np.newaxis],
                        axis=-1,
                    ).squeeze(
                        -1
                    )  # (num_agents, timesteps, batch_size,)

                    target = (
                        _rewards[:, :-1]
                        + (1 - _dones[:, :-1]) * config["GAMMA"] * q_next[:, 1:]
                    )

                    chosen_action_q_vals = chosen_action_q_vals[:, :-1]
                    loss = jnp.mean(
                        (chosen_action_q_vals - jax.lax.stop_gradient(target)) ** 2
                    )

                    return loss, chosen_action_q_vals.mean()

                (loss, qvals), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                    train_state.params
                )
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(
                    grad_steps=train_state.grad_steps + 1,
                )
                return (train_state, rng), (loss, qvals)

            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                buffer.can_sample(buffer_state)
            ) & (  # enough experience in buffer
                train_state.timesteps > config["LEARNING_STARTS"]
            )
            (train_state, rng), (loss, qvals) = jax.lax.cond(
                is_learn_time,
                lambda train_state, rng: jax.lax.scan(
                    _learn_phase, (train_state, rng), None, config["NUM_EPOCHS"]
                ),
                lambda train_state, rng: (
                    (train_state, rng),
                    (
                        jnp.zeros(config["NUM_EPOCHS"]),
                        jnp.zeros(config["NUM_EPOCHS"]),
                    ),
                ),  # do nothing
                train_state,
                _rng,
            )

            # update target network
            train_state = jax.lax.cond(
                train_state.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda train_state: train_state.replace(
                    target_network_params=optax.incremental_update(
                        train_state.params,
                        train_state.target_network_params,
                        config["TAU"],
                    )
                ),
                lambda train_state: train_state,
                operand=train_state,
            )

            # UPDATE METRICS
            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "loss": loss.mean(),
                "qvals": qvals.mean(),
            }
            metrics.update(jax.tree.map(lambda x: x.mean(), infos))
            if config.get("LOG_AGENTS_SEPARATELY", False):
                for i, a in enumerate(env.agents):
                    m = jax.tree.map(
                        lambda x: x[..., i].mean(),
                        infos,
                    )
                    m = {k + f"_{a}": v for k, v in m.items()}
                    metrics.update(m)

            # update the test metrics
            if config.get("TEST_DURING_TRAINING", True):
                rng, _rng = jax.random.split(rng)
                test_state = jax.lax.cond(
                    train_state.n_updates
                    % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"])
                    == 0,
                    lambda _: get_greedy_metrics(_rng, train_state),
                    lambda _: test_state,
                    operand=None,
                )
                metrics.update({"test_" + k: v for k, v in test_state.items()})

            # report on wandb if required
            #if config["WANDB_MODE"] != "disabled":

            #    def callback(metrics, original_seed):
            #        if config.get('WANDB_LOG_ALL_SEEDS', False):
            #            metrics.update(
            #                {f"rng{int(original_seed)}/{k}": v for k, v in metrics.items()}
            #            )
            #        wandb.log(metrics)

            #    jax.debug.callback(callback, metrics, original_seed)

            runner_state = (train_state, buffer_state, test_state, rng)
            print("Completed update step")

            return runner_state, None

        def get_greedy_metrics(rng, train_state):
            """Help function to test greedy policy during training"""
            if not config.get("TEST_DURING_TRAINING", True):
                return None
            params = train_state.params
            def _greedy_env_step(step_state, unused):
                params, env_state, last_obs, last_dones, hstate, rng = step_state
                rng, key_s = jax.random.split(rng)
                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]
                hstate, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                    params,
                    hstate,
                    _obs,
                    _dones,
                )
                q_vals = q_vals.squeeze(axis=1)
                valid_actions = test_env.get_valid_actions(env_state.env_state)
                actions = get_greedy_actions(q_vals, batchify(valid_actions))
                actions = unbatchify(actions)
                obs, env_state, rewards, dones, infos = test_env.batch_step(
                    key_s, env_state, actions
                )
                step_state = (params, env_state, obs, dones, hstate, rng)
                return step_state, (rewards, dones, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            init_dones = {
                agent: jnp.zeros((config["TEST_NUM_ENVS"]), dtype=bool)
                for agent in env.agents + ["__all__"]
            }
            rng, _rng = jax.random.split(rng)
            hstate = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], len(env.agents), config["TEST_NUM_ENVS"]
            )  # (n_agents*n_envs, hs_size)
            step_state = (
                params,
                env_state,
                init_obs,
                init_dones,
                hstate,
                _rng,
            )
            step_state, (rewards, dones, infos) = jax.lax.scan(
                _greedy_env_step, step_state, None, config["TEST_NUM_STEPS"]
            )
            if config.get("LOG_AGENTS_SEPARATELY", False):
                metrics = {}
                for i, a in enumerate(env.agents):
                    m = jax.tree.map(
                        lambda x: jnp.nanmean(
                            jnp.where(
                                infos["returned_episode"][..., i],
                                x[..., i],
                                jnp.nan,
                            )
                        ),
                        infos,
                    )
                    m = {k + f"_{a}": v for k, v in m.items()}
                    metrics.update(m)
            else:
                metrics = jax.tree.map(
                    lambda x: jnp.nanmean(
                        jnp.where(
                            infos["returned_episode"],
                            x,
                            jnp.nan,
                        )
                    ),
                    infos,
                )
            return metrics

        rng, _rng = jax.random.split(rng)
        test_state = get_greedy_metrics(_rng, train_state)

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, buffer_state, test_state, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train


def env_from_config(config):
    env_name = config["ENV_NAME"]
    # smax init neeeds a scenario
    if "smax" in env_name.lower():
        config["ENV_KWARGS"]["scenario"] = map_name_to_scenario(config["MAP_NAME"])
        env_name = f"{config['ENV_NAME']}_{config['MAP_NAME']}"
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = SMAXLogWrapper(env)
    # overcooked needs a layout
    elif "overcooked" in env_name.lower():
        env_name = f"{config['ENV_NAME']}_{config['ENV_KWARGS']['layout']}"
        config["ENV_KWARGS"]["layout"] = overcooked_layouts[
            config["ENV_KWARGS"]["layout"]
        ]
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = LogWrapper(env)
    elif "mpe" in env_name.lower():
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = MPELogWrapper(env)
    else:
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = LogWrapper(env)
    return env, env_name


def single_run(config):

    #config = {**config, **config["alg"]}  # merge the alg config with the main config
    #print("Config:\n", OmegaConf.to_yaml(config))

    alg_name = config.get("ALG_NAME", "iql_rnn")
    #env, env_name= env_from_config(copy.deepcopy(config))
    config["ENV_KWARGS"]["scenario"] = map_name_to_scenario(config["MAP_NAME"])
    env_name = f"{config['ENV_NAME']}_{config['MAP_NAME']}"
    env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = SMAXLogWrapper(env)

    #wandb.init(
    #    entity=config["ENTITY"],
    #    project=config["PROJECT"],
    #    tags=[
    #        alg_name.upper(),
    #        env_name.upper(),
    #        f"jax_{jax.__version__}",
    #    ],
    #    name=f"{alg_name}_{env_name}",
    #    config=config,
    #    mode=config["WANDB_MODE"],
    #)

    rng = jax.random.PRNGKey(config["SEED"])

    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config, env)))
    outs = jax.block_until_ready(train_vjit(rngs))

    print(len(outs))

    # save params
    #if config.get("SAVE_PATH", None) is not None:
    #    from jaxmarl.wrappers.baselines import save_params

    #    model_state = outs["runner_state"][0]
    #    save_dir = os.path.join(config["SAVE_PATH"], env_name)
    #    os.makedirs(save_dir, exist_ok=True)
    #    OmegaConf.save(
    #        config,
    #        os.path.join(
    #            save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml'
    #        ),
    #    )

    #    for i, rng in enumerate(rngs):
    #        params = jax.tree.map(lambda x: x[i], model_state.params)
    #        save_path = os.path.join(
    #            save_dir,
    #            f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}.safetensors',
    #        )
    #        save_params(params, save_path)


#def tune(default_config):
#    """Hyperparameter sweep with wandb."""

#    default_config = {**default_config, **default_config["alg"]}  # merge the alg config with the main config
#    env_name = default_config["ENV_NAME"]
#    alg_name = default_config.get("ALG_NAME", "iql_rnn") 
#    env, env_name = env_from_config(default_config)

#    def wrapped_make_train():
#        wandb.init(project=default_config["PROJECT"])

        # update the default params
#        config = copy.deepcopy(default_config)
#        for k, v in dict(wandb.config).items():
#            config[k] = v

#        print("running experiment with params:", config)

#        rng = jax.random.PRNGKey(config["SEED"])
#        rngs = jax.random.split(rng, config["NUM_SEEDS"])
#        train_vjit = jax.jit(jax.vmap(make_train(config, env)))
#        outs = jax.block_until_ready(train_vjit(rngs))

#    sweep_config = {
#        "name": f"{alg_name}_{env_name}",
#        "method": "bayes",
#        "metric": {
#            "name": "test_returned_episode_returns",
#            "goal": "maximize",
#        },
#        "parameters": {
#            "LR": {
#                "values": [
#                    0.005,
#                    0.001,
#                    0.0005,
#                    0.0001,
#                    0.00005,
#                ]
#            },
#            "NUM_ENVS": {"values": [8, 32, 64, 128]},
#        },
#    }

#    wandb.login()
#    sweep_id = wandb.sweep(
#        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
#    )
#    wandb.agent(sweep_id, wrapped_make_train, count=300)


#@hydra.main(version_base=None, config_path="./config", config_name="config")
#def main(config):
#    config = OmegaConf.to_container(config)
#    print("Config:\n", OmegaConf.to_yaml(config))
#    if config["HYP_TUNE"]:
#        tune(config)
#    else:
#        single_run(config)


# -----------------------------
# Visualize recurrent IPPO policy
# -----------------------------

def batchify(x: dict):
    return jnp.stack([x[agent] for agent in env.agents], axis=0)

def unbatchify(x: jnp.ndarray):
    return {agent: x[i] for i, agent in enumerate(env.agents)}

def preprocess_obs_with_id(obs_dict, env):
    """Simulate CTRolloutManager's preprocessing by adding one-hot agent IDs."""
    new_obs_dict = {}
    num_agents = len(env.agents)
    for i, agent in enumerate(env.agents):
        obs = obs_dict[agent].flatten()
        one_hot = jax.nn.one_hot(i, num_classes=num_agents)
        new_obs_dict[agent] = jnp.concatenate([obs, one_hot])
    return new_obs_dict


def visualize_recurrent_policy(trained_params, env, config):
    rng = jax.random.PRNGKey(config["SEED"])
    rng, reset_rng = jax.random.split(rng)
    #wrapped_env = CTRolloutManager(env, batch_size=1)

    # Create policy network
    #network = RNNQNetwork(
    #    action_dim=wrapped_env.max_action_space,
    #    hidden_dim=config["HIDDEN_SIZE"],
    #)
    network = RNNQNetwork(
        action_dim=env.action_space(env.agents[0]).n,
        hidden_dim=config["HIDDEN_SIZE"],
    )
    
    # Reset environment
    #obs, env_state = wrapped_env.batch_reset(reset_rng)
    obs, env_state = env.reset(reset_rng)
    #dones = {
    #    agent: jnp.zeros((1), dtype=bool)
    #    for agent in env.agents + ["__all__"]
    #}
    dones = {agent: jnp.array(False) for agent in env.agents}
    hstate = ScannedRNN.initialize_carry(
        config["HIDDEN_SIZE"], len(env.agents), 1
    )
    
    # Collect all environment states
    returns = {agent: 0.0 for agent in env.agents}
    state_seq = []
    max_steps = config["NUM_STEPS"]

    for step in range(max_steps):
        # Compute Q-values
        # Prepare inputs for network
        obs = preprocess_obs_with_id(obs, env)
        _obs = batchify(obs)         # (num_agents, obs_dim)
        _obs = _obs[:, None, :]                      # (num_agents, 1, obs_dim)

        #_dones = batchify(dones)    # (num_agents,)
        #_dones = _dones[:, None]                     # (num_agents, 1)
        _dones = jnp.stack([jnp.array([dones[agent]]) for agent in env.agents])  # shape (num_agents, 1)
        _dones = jnp.expand_dims(_dones, axis=-1)  # from (3, 1) to (3, 1, 1)

        #print("_obs.shape:", _obs.shape)
        #print("_dones.shape:", _dones.shape)
        print("hstate.shape:", hstate.shape)

        def apply_fn(h, o, d):
            return network.apply(trained_params, h, o, d)

        hstate, q_vals = jax.vmap(apply_fn, in_axes=(0, 0, 0))(
            hstate,
            _obs,
            _dones,
        )
        print("hstate.shape:", hstate.shape)

        #hstate = hstate[:, None, :]  # Already in (num_agents, hidden_dim)
        q_vals = q_vals.squeeze(axis=1)  # (num_agents, num_envs, num_actions) remove the time dim
        #print("q_vals.shape", q_vals.shape)
        
        actions = {}
        #avail_actions = wrapped_env.get_valid_actions(env_state.env_state)
        avail_actions = env.get_avail_actions(env_state.env_state)

        for i, agent in enumerate(env.agents):
            avail_agent = avail_actions[agent][None, None, :]  # shape (1, 1, n_actions)
            #print("avail_agent.shape", avail_agent.shape)
            
            unavail_actions = 1 - avail_agent  # shape (1, 1, n_actions)
            
            # Select Q-values for this agent only
            q_agent = q_vals[i][None, None, :]  # shape (1, 1, n_actions)
            q_masked = q_agent - (unavail_actions * 1e10)

            action = jnp.argmax(q_masked, axis=-1)  # shape (1, 1)
            action = action.squeeze()               # scalar
            #print("action.shape", action.shape)

            # Wrap in array with batch dim
            actions[agent] = int(action)    # shape (1,)
        
        rng, rng_s = jax.random.split(rng)
        state_seq.append((rng_s, env_state.env_state, actions))

        # Step environment

        # Batch the actions dict
        # Original actions: {'ally_0': 4, 'ally_1': 4, 'ally_2': 4}
        #actions = {k: jnp.array([v]) for k, v in actions.items()}

        #obs, env_state, rewards, dones, infos = wrapped_env.batch_step(
        #    rng_s, env_state, actions
        #)
        obs, env_state, rewards, dones, infos = env.step(rng_s, env_state, actions)
        returns = {a: returns[a] + rewards[a] for a in env.agents}
        
        if dones["__all__"]:
            break

    # Visualization
    print("Returns:", returns)

    viz = SMAXVisualizer(env, state_seq)
    viz.animate(view=False, save_fname="trained_iql_rnn.gif")

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    config = {
        # valid for iql, vdn, qmix
        "TOTAL_TIMESTEPS": 1e7,
        "NUM_ENVS": 128, #16,
        "NUM_STEPS": 128,
        "BUFFER_SIZE": 5000,
        "BUFFER_BATCH_SIZE": 32,
        "HIDDEN_SIZE": 512,
        "MIXER_EMBEDDING_DIM": 64,
        "MIXER_HYPERNET_HIDDEN_DIM": 256,
        "MIXER_INIT_SCALE": 0.001,
        "EPS_START": 1.0,
        "EPS_FINISH": 0.05,
        "EPS_DECAY": 0.1, # percentage of updates
        "MAX_GRAD_NORM": 10,
        "TARGET_UPDATE_INTERVAL": 10,
        "TAU": 1.,
        "NUM_EPOCHS": 8,
        "LR": 0.00005,
        "LEARNING_STARTS": 10000, # timesteps
        "LR_LINEAR_DECAY": False,
        "GAMMA": 0.99,
        "REW_SCALE": 10., # scale the reward to the original scale of SMAC

        # ENV
        "ENV_NAME": "HeuristicEnemySMAX",
        #"MAP_NAME": "3s_vs_5z",
        "MAP_NAME": "2s3z",
        "ENV_KWARGS": {
            "see_enemy_actions": True,
            "walls_cause_death": True,
            "attack_mode": "closest",
        },

        "NUM_SEEDS": 1, # number of vmapped seeds
        "SEED": 0,

        "HYP_TUNE": False, # perform hyp tune

        # evaluate
        "TEST_DURING_TRAINING": False, #True,
        "TEST_INTERVAL": 0.05, # as a fraction of updates, i.e. log every 5% of training process
        "TEST_NUM_STEPS": 128,
        "TEST_NUM_ENVS": 512, # number of episodes to average over, can affect performance
    }

    # Setup env
    scenario = map_name_to_scenario(config["MAP_NAME"])
    env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])
    env = SMAXLogWrapper(env)

    # Prepare RNG
    rng = jax.random.PRNGKey(config["SEED"])
    #rngs = jax.random.split(rng, config["NUM_SEEDS"])

    # Get the training function from make_train
    train_fn = make_train(config, env)

    # JIT compile the train function
    train_jit = jax.jit(train_fn, device=jax.devices()[0])
    #train_vjit = jax.jit(jax.vmap(make_train(config, env)))
    
    # Run training and get output
    output = train_jit(rng)
    #outs = jax.block_until_ready(train_vjit(rngs))

    # Extract trained parameters from output
    trained_params = output["runner_state"][0].params

    # Visualize policy
    visualize_recurrent_policy(trained_params, env, config)

    #single_run(config)