#!/usr/bin/env python3
import os.path as osp
import argparse
from baselines.common.cmd_util import mujoco_arg_parser
from baselines import bench, logger

def train(env_id, num_timesteps, seed, network, r_ex_coef, r_in_coef, lr, reward_freq,
          begin_iter, model_train_num, K_model_num, regularize, selection_type):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2_ensemble
    from baselines.ppo2.policies import MlpPolicy
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()
    def make_env():
        env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir())
        return env
    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)

    if network == 'mlp':
        network = MlpPolicy
    else:
        raise NotImplementedError

    ppo2_ensemble.learn(network=network, env=env, nsteps=2048, nminibatches=32,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=0.0,
        lr=lr,
        cliprange=0.2,
        total_timesteps=num_timesteps,
        r_ex_coef=r_ex_coef,
        beta=r_in_coef,
        reward_freq=reward_freq,
        begin_iter=begin_iter,
        model_train=model_train_num,
        K=K_model_num,
        alpha=regularize,
        index_type=selection_type)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='Environment ID', default='Walker2d-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--network', help='Policy architecture', default='mlp')
    parser.add_argument('--num-timesteps', type=int, default=int(1E6))
    parser.add_argument('--r-ex-coef', type=float, default=0)
    parser.add_argument('--r-in-coef', type=float, default=1)
    parser.add_argument('--lr', type=float, default=3E-4)
    parser.add_argument('--reward-freq', type=int, default=40)
    parser.add_argument('--begin-iter', type=int, default=40)
    parser.add_argument('--model-train-num', type=int, default=4)
    parser.add_argument('--K-model-num', type=int, default=1)
    parser.add_argument('--regularize', type=float, default=0.01)
    parser.add_argument('--selection-type', choices=['min', 'max', 'avg', 'ens'], default='min')
    args = parser.parse_args()
    logger.configure()
    logger.configure(osp.join(osp.abspath(osp.dirname(__file__)), 'Results_ensemble', 'begin_iter=' + str(args.begin_iter),
                              args.env + '_freq' + str(args.reward_freq),
                              'r_ex_coef=' + str(args.r_ex_coef) + ', r_in_coef=' + str(args.r_in_coef)
                              + ', K_model_num=' + str(args.K_model_num) + ', model_train_num=' + str(args.model_train_num)
                              + ', regularize=' + str(args.regularize) + ', selection_type=' + str(args.selection_type),
                              'iter' + str(args.seed)))
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, network=args.network,
          r_ex_coef=args.r_ex_coef, r_in_coef=args.r_in_coef,
          lr=args.lr, reward_freq=args.reward_freq,
          begin_iter=args.begin_iter, model_train_num=args.model_train_num, K_model_num=args.K_model_num,
          regularize=args.regularize, selection_type=args.selection_type)


if __name__ == '__main__':
    main()
