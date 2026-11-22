"""
Env builders. They are top-level functions so they can be pickled and called
inside the physics process (AsyncEnv uses the "spawn" start method).
"""
LOW_DIM_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]


def env_from_ckpt(ckpt_path, render_offscreen=False):
    """Same env (incl. frame stacking wrapper) that the policy was trained for."""
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils

    ckpt = FileUtils.maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt)
    ObsUtils.initialize_obs_utils_with_config(config)
    env, _ = FileUtils.env_from_checkpoint(ckpt_dict=ckpt, render=False,
                                           render_offscreen=render_offscreen, verbose=False)
    return env


def env_from_dataset(dataset_path, render_offscreen=False):
    """Lift env from a robomimic dataset (used with a random policy for smoke tests)."""
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils

    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": LOW_DIM_KEYS}})
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    return EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False,
                                             render_offscreen=render_offscreen,
                                             use_image_obs=False)


def obs_spec(obs):
    """{key: shape} of an observation dict, for AsyncEnv's shared-memory layout."""
    import numpy as np
    return {k: tuple(np.asarray(v).shape) for k, v in obs.items()}
