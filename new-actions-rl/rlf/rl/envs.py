import os

import gym
import numpy as np
import torch
from gym.spaces.box import Box

from rlf.baselines.monitor import Monitor
from rlf.baselines.common.atari_wrappers import make_atari, wrap_deepmind
from rlf.baselines.vec_env import VecEnvWrapper
from rlf.baselines.vec_env.dummy_vec_env import DummyVecEnv
from rlf.baselines.vec_env.shmem_vec_env import ShmemVecEnv
from rlf.baselines.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_

# Include our custom environments
import envs.gym_minigrid

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass


def make_env(env_id, seed, rank, log_dir, allow_early_resets, trans_fn,
        set_eval):
    def _thunk():
        if env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)

        env.seed(seed + rank)

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        env = trans_fn(env, set_eval)

        # Dont need this garbage
        if log_dir is not None:
            env = Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets)

        if is_atari:
            if len(env.observation_space.shape) == 3:
                env = wrap_deepmind(env)
        #elif len(env.observation_space.shape) == 3:
        #    raise NotImplementedError(
        #        "CNN models work only for atari,\n"
        #        "please use a custom wrapper for a custom pixel input env.\n"
        #        "See wrap_deepmind for an example.")

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3, 5]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk

env_names = ["CreateLevelPush-v0", "CreateLevelObstacle-v0", "CreateLevelLadder-v0"]

def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  gamma,
                  log_dir,
                  device,
                  allow_early_resets,
                  trans_fn,
                  args,
                  num_frame_stack=None,
                  set_eval=False):

    if args.multitask and not set_eval:
        return make_vec_envs_for_multitask(env_names, seed, num_processes, gamma, log_dir, device, allow_early_resets, trans_fn, args, num_frame_stack, set_eval)

    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets, trans_fn,
            set_eval)
        for i in range(num_processes)
    ]

    print(f"make_env completed: {len(envs)} envs created")

    if len(envs) > 1:
        envs = ShmemVecEnv(envs, context='fork')
        # envs = ShmemVecEnv(envs, context='spawn')
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, ret=False)
        else:
            envs = VecNormalize(envs, gamma=gamma)

    envs = VecPyTorch(envs, device)
    print(f"VecPyTorch finished")

    if env_name.startswith('Create'):
        num_frame_stack = 3

    if num_frame_stack is not None and not args.no_frame_stack:
        print(f"apply VecPyTorchFrameStack: {num_frame_stack}")
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
        print(f"apply VecPyTorchFrameStack completed!")
    elif len(envs.observation_space.shape) == 3 and not args.no_frame_stack:
        envs = VecPyTorchFrameStack(envs, 4, device)

    return envs

def make_vec_envs_for_multitask(env_names,
                  seed,
                  num_processes,
                  gamma,
                  log_dir,
                  device,
                  allow_early_resets,
                  trans_fn,
                  args,
                  num_frame_stack=None,
                  set_eval=False,
                  env_names_input=None):

    candidate_env_names = args.env_names if len(args.env_names) > 1 else env_names
    envs = [
        [ make_env(env_name, seed, i, log_dir, allow_early_resets, trans_fn, set_eval) for i in range(num_processes)]
        for env_name in candidate_env_names
    ]

    print(f"make_env_multitask completed: {len(envs)} tasks {len(envs[0])} envs created")

    for i in range(len(envs)):
        env_name = candidate_env_names[i]
        if len(envs[i]) > 1:
            envs[i] = ShmemVecEnv(envs[i], context='fork')
            # envs[i] = ShmemVecEnv(envs[i], context='spawn')
        else:
            envs[i] = DummyVecEnv(envs[i])

        if len(envs[i].observation_space.shape) == 1:
            if gamma is None:
                envs[i] = VecNormalize(envs[i], ret=False)
            else:
                envs[i] = VecNormalize(envs[i], gamma=gamma)

        envs[i] = VecPyTorch(envs[i], device)
        print(f"finished VecPyTorch for env task {i}")

        if env_name.startswith('Create'):
            num_frame_stack = 3

        if num_frame_stack is not None and not args.no_frame_stack:
            print(f"apply VecPyTorchFrameStack: {num_frame_stack}")
            envs[i] = VecPyTorchFrameStack(envs[i], num_frame_stack, device)
            print(f"apply VecPyTorchFrameStack completed!")
        elif len(envs[i].observation_space.shape) == 3 and not args.no_frame_stack:
            envs[i] = VecPyTorchFrameStack(envs[i], 4, device)

    return envs

# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def get_aval(self):
        return self.env.get_aval()


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation

    def get_aval(self):
        return self.env.get_aval()



class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)
    def get_aval(self):
        return self.env.get_aval()


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])
    def get_aval(self):
        return self.env.get_aval()


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.tensor(obs).to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.Tensor(obs)
        reward = torch.Tensor(reward).unsqueeze(dim=1)
        # Reward is sometimes a Double. Observation is considered to always be
        # float32
        reward = reward.type(obs.type())
        obs = obs.to(self.device)
        return obs, reward, done, info
    def get_aval(self):
        return self.venv.get_aval()


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) /
                          np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def get_aval(self):
        return self.venv.get_aval()


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()

        a = self.stacked_obs.clone().detach()
        # self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=-1)
        self.stacked_obs[:, :-self.shape_dim0] = a[:, self.shape_dim0:]
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()

    def get_aval(self):
        return self.venv.get_aval()
