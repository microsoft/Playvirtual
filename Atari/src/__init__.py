# --------------------------------------------------------
# This code is borrowed from https://github.com/mila-iqia/spr/blob/release/src/__init__.py
# --------------------------------------------------------

from gym.envs.registration import register

register(
    id='atari-v0',
    entry_point='src.envs:AtariEnv',
)