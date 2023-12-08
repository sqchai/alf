# Copyright (c) 2023 Horizon Robotics and Hobot Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A tool for checking whether two revisions of code generate the same checkpoints.

This tool will run the training with reduced num_env/mini_batch_size/iterations/initial_collect_steps
and compare the checkpoints generated by the two runs.

Example 1:

``bash
python tools/compare_training.py --conf alf/examples/sac_cart_pole_conf.py --rev1 5514d8a6
```
 Example 2:

 ```bash
 python tools/compare_training.py --conf alf/examples/sac_cart_pole_conf.py --rev1 5514d8a6 --rev2 3dcaf061
 ```

"""

from absl import app
from absl import logging
from absl import flags
from pathlib import Path
import subprocess
import tempfile
from alf.utils.common import alf_root
from alf.utils.git_utils import get_revision, get_diff, _exec
from alf.bin.train_play_test import run_cmd

flags.DEFINE_string(
    "conf", None, help="The config file for training", required=True)
flags.DEFINE_string("rev1", None, help="The first revision.", required=True)
flags.DEFINE_string(
    "rev2",
    None,
    help=
    "The second revision. If not provided, the current revision will be used",
    required=False)
flags.DEFINE_integer(
    "iterations", 5, help="The number of iterations to run", required=False)
flags.DEFINE_integer(
    "num_envs", 10, help="The number of environments to run", required=False)
flags.DEFINE_integer(
    "mini_batch_size", 256, help="Minibatch size", required=False)
flags.DEFINE_integer(
    "initial_collect_steps",
    10,
    help="The number of steps to collect before training",
    required=False)
flags.DEFINE_integer(
    "unroll_length",
    10,
    help=" number of time steps each environment proceeds per iteration.",
    required=False)

FLAGS = flags.FLAGS


def run_train(conf, root_dir, rev1):
    cmd = ["git", "checkout", rev1]
    run_cmd(cmd=cmd, cwd='.')

    cmd = [
        'python3',
        '-m',
        'alf.bin.train',
        '--nostore_snapshot',
        '--root_dir=%s' % root_dir,
        '--conf=%s' % conf,
        '--conf_param=TrainerConfig.confirm_checkpoint_upon_crash=0',
        '--conf_param=TrainerConfig.random_seed=1',
        '--conf_param=TrainerConfig.num_checkpoints=1',
        '--conf_param=TrainerConfig.unroll_length=%s' % FLAGS.unroll_length,
        '--conf_param=TrainerConfig.num_iterations=%s' % FLAGS.iterations,
        '--conf_param=TrainerConfig.mini_batch_size=%s' %
        FLAGS.mini_batch_size,
        '--conf_param=TrainerConfig.initial_collect_steps=%s' %
        FLAGS.initial_collect_steps,
        '--conf_param=create_environment.num_parallel_environments=%s' %
        FLAGS.num_envs,
        '--conf_param=create_environment.batch_size_per_env=2',
        '--conf_param=TrainerConfig.num_env_steps=0',
    ]
    run_cmd(cmd=cmd, cwd='.')


def get_current_branch(module_root):
    return _exec("git rev-parse --abbrev-ref HEAD", module_root)


def switch_branch(module_root, branch):
    return _exec("git checkout %s" % branch, module_root)


def main(_):
    if FLAGS.initial_collect_steps > (
            FLAGS.iterations - 1) * FLAGS.unroll_length * FLAGS.num_envs:
        logging.error(
            "initial_collect_steps should be <= (num_iterations - 1) * unroll_length * num_parallel_environments"
        )
        exit(1)

    repo_root = Path(alf_root())
    if get_diff(repo_root):
        logging.error(
            "You need to commit all changes before running this script")
        exit(1)
    current_branch = get_current_branch(repo_root)
    if FLAGS.rev2 is None:
        FLAGS.rev2 = get_revision(repo_root)

    try:
        root1 = tempfile.TemporaryDirectory()
        root2 = tempfile.TemporaryDirectory()
        root_dir1 = root1.name
        root_dir2 = root2.name
        run_train(FLAGS.conf, root_dir1, FLAGS.rev1)
        run_train(FLAGS.conf, root_dir2, FLAGS.rev2)
    finally:
        switch_branch(repo_root, current_branch)

    cmd = ' '.join([
        "diff", "-r", root_dir1 + "/train/algorithm",
        root_dir2 + "/train/algorithm"
    ])
    diff = _exec(cmd, ".")
    if diff:
        logging.error("The checkpoints from the two runs are different.")
        logging.error(diff)
        exit(1)
    else:
        logging.info("The checkpoints from the two runs are same.")


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
