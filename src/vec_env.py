import multiprocessing as mp
import numpy as np
import time
from typing import List, Tuple, Dict, Any, Optional, Callable

from world import World
from configs import WorldConfig

# Define CloudpickleWrapper for better pickling across processes
try:
    import cloudpickle
    def _worker_shared_mem(remote, parent_remote, env_fn_wrapper):
        """Worker function that uses cloudpickle for env_fn_wrapper."""
        parent_remote.close()
        env = env_fn_wrapper.var() # Cloudpickledeserializes the env_fn_wrapper
        try:
            while True:
                cmd, data = remote.recv()
                if cmd == 'step':
                    # Assuming env.step takes action and returns obs, reward, done, info
                    # World.step modifies internal state and returns nothing directly
                    # Need to adapt World or this worker
                    action = data
                    terminal_step = False # How to determine this? Maybe pass max_steps?
                    # Let's assume the VecEnv user tracks steps and tells worker when it's terminal
                    # Or, pass 'training' flag if World.step uses it for reward calc.
                    env.step(action, training=True, terminal_step=data.get('terminal_step', False) if isinstance(data, dict) else False) # Adapt based on actual step signature needs
                    obs = env.encode_state()
                    reward = env.reward
                    done = env.done
                    info = {'error_dist': env.error_dist} # Add relevant info

                    if done:
                        # Auto-reset environment if done
                        # print(f"Worker {mp.current_process().pid}: Episode done. Resetting.") # Debug
                        env.reset()
                        obs = env.encode_state() # Get observation after reset

                    remote.send((obs, reward, done, info))
                elif cmd == 'reset':
                    env.reset()
                    obs = env.encode_state()
                    remote.send(obs)
                elif cmd == 'close':
                    remote.close()
                    break
                elif cmd == 'get_spaces':
                    # Placeholder if needed for observation/action spaces
                    remote.send((None, None))
                else:
                    raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except KeyboardInterrupt:
            print('SubprocVecEnv worker: got KeyboardInterrupt')
        except EOFError:
             print(f"Worker {mp.current_process().pid}: Pipe closed unexpectedly.")
        finally:
            env.close() # Assuming World has a close method (optional)

    class CloudpickleWrapper:
        """Uses cloudpickle to serialize and deserialize objects."""
        def __init__(self, var):
            self.var = var
        def __getstate__(self):
            return cloudpickle.dumps(self.var)
        def __setstate__(self, obs):
            self.var = cloudpickle.loads(obs)

except ImportError:
    cloudpickle = None
    print("Warning: cloudpickle not installed. Falling back to default pickle for VecEnv.")
    # Define _worker using standard pickle (less robust)
    def _worker(remote, parent_remote, env_fn):
        parent_remote.close()
        env = env_fn() # Call the function to create env
        try:
            while True:
                cmd, data = remote.recv()
                if cmd == 'step':
                    action = data.get('action') if isinstance(data, dict) else data
                    terminal_step = data.get('terminal_step', False) if isinstance(data, dict) else False
                    env.step(action, training=True, terminal_step=terminal_step)
                    obs = env.encode_state()
                    reward = env.reward
                    done = env.done
                    info = {'error_dist': env.error_dist}
                    if done:
                        env.reset()
                        obs = env.encode_state()
                    remote.send((obs, reward, done, info))
                elif cmd == 'reset':
                    env.reset()
                    obs = env.encode_state()
                    remote.send(obs)
                elif cmd == 'close':
                    remote.close()
                    break
                elif cmd == 'get_spaces':
                    remote.send((None, None)) # Placeholder
                else:
                    raise NotImplementedError
        except KeyboardInterrupt: print('SubprocVecEnv worker: got KeyboardInterrupt')
        except EOFError: print(f"Worker {mp.current_process().pid}: Pipe closed unexpectedly.")
        finally: pass # No close assumed


def make_env(world_config: WorldConfig, seed: int = 0):
    """
    Utility function for multiprocessed envs.

    :param world_config: The configuration for the World.
    :param seed: The initial seed for the environment.
    """
    def _init():
        # print(f"Creating env in process {mp.current_process().pid} with seed {seed}") # Debug
        # Note: Seeding needs careful implementation within World/random if needed
        # random.seed(seed)
        # np.random.seed(seed)
        env = World(world_config=world_config)
        # Consider adding env.seed(seed) method if precise seeding is required
        return env
    return _init

class SubprocVecEnv:
    """
    Creates a multiprocess vectorized wrapper for multiple environments.

    :param env_fns: A list of functions that create the environments.
    """
    def __init__(self, env_fns: List[Callable[[], World]]):
        self.num_envs = len(env_fns)
        self.waiting = False
        self.closed = False
        ctx = mp.get_context('spawn') # 'spawn' is generally safer

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn) if cloudpickle else env_fn)
            process = ctx.Process(target=_worker_shared_mem if cloudpickle else _worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close() # Close worker end in parent process

        # Initialize observation buffer shape (based on trajectory state)
        self.remotes[0].send(('reset', None))
        initial_obs_dict = self.remotes[0].recv()
        self._obs_shape = initial_obs_dict['full_trajectory'].shape
        self._feature_dim = initial_obs_dict['full_trajectory'].shape[1]
        self._traj_len = initial_obs_dict['full_trajectory'].shape[0]

    def step_async(self, actions: np.ndarray):
        """ Send actions to the environments """
        if self.waiting:
            raise AlreadySteppingError("Already stepping in the environment. Call step_wait first.")
        if self.closed:
            raise ClosedEnvironmentError("Trying to step closed environments")

        for remote, action in zip(self.remotes, actions):
             remote.send(('step', action)) # Send only the action
        self.waiting = True

    def step_wait(self) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """ Wait for the results from the environments """
        if not self.waiting:
            raise NotSteppingError("Calling step_wait before step_async. Call step_async first.")
        if self.closed:
            raise ClosedEnvironmentError("Trying to step closed environments")

        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs_list, rewards, dones, infos = zip(*results)

        # obs_list contains dictionaries {'basic_state': ..., 'full_trajectory': ..., 'estimator_state': ...}
        return list(obs_list), np.stack(rewards), np.stack(dones), list(infos)

    def step(self, actions: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """ Step the environments with the given actions """
        self.step_async(actions)
        return self.step_wait()

    def reset(self) -> List[Dict[str, Any]]:
        """ Reset all environments """
        if self.closed:
             raise ClosedEnvironmentError("Trying to reset closed environments")
        for remote in self.remotes:
            remote.send(('reset', None))
        obs_list = [remote.recv() for remote in self.remotes]
        return list(obs_list) # Return list of state dictionaries

    def close(self):
        """ Close all environments """
        if self.closed:
            return
        if self.waiting:
            # Wait for pending steps to finish before closing
            try:
                results = [remote.recv() for remote in self.remotes]
            except EOFError:
                print("Warning: EOFError received while waiting during close. Some processes might have already terminated.")
            self.waiting = False

        for remote in self.remotes:
             try:
                  remote.send(('close', None))
                  remote.close() # Close parent end of pipe
             except BrokenPipeError:
                  print("Warning: BrokenPipeError during close. Process might have already terminated.")

        for process in self.processes:
             process.join()
        self.closed = True

    def __len__(self) -> int:
        return self.num_envs

class AlreadySteppingError(Exception):
    """Raised when step_async is called before step_wait."""
    pass

class NotSteppingError(Exception):
    """Raised when step_wait is called before step_async."""
    pass

class ClosedEnvironmentError(Exception):
    """Raised when trying to interact with closed environments."""
    pass
