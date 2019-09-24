import tensorflow as tf
from baselines.common import tf_util_ensemble
from baselines.a2c.utils import fc
from baselines.common.distributions_ensemble import make_pdtype
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.tf_util_ensemble import adjust_shape
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.models import get_network_builder
import numpy as np

import gym
from gym.spaces import Discrete, Box, MultiDiscrete

import baselines.common.tf_util_ensemble as U


class DynamicsWithValue(object):
    """
    Encapsulates fields and methods for RL dynamics and value function estimation with shared parameters
    """

    def __init__(self, env, observations, latent, sess=None, index=None, **tensors):
        """
        Parameters:
        ----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        latent          latent state from which dynamics distribution parameters should be inferred

        vf_latent       latent state from which value function should be inferred (if None, then latent is used)

        sess            tensorflow session to run calculations in (if None, default session is used)

        **tensors       tensorflow tensors for additional attributes such as state or mask

        """

        self.X = observations
        self.state = tf.constant([])
        self.initial_state = None
        self.__dict__.update(tensors)

        # sequence_length = None
        # ob_size = list(env.observation_space.shape)
        # ac_size = list(env.action_space.shape)
        #
        # input_size = (np.add(ob_size, ac_size)).tolist()
        #
        # input_dyn = U.get_placeholder(name="input_dyn1", dtype=tf.float32,
        #                        shape=[sequence_length] + input_size )

        # vf_latent = vf_latent if vf_latent is not None else latent
        #
        # vf_latent = tf.layers.flatten(vf_latent)
        latent = tf.layers.flatten(latent)

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(env.observation_space)
        # self.pdtype = make_pdtype(env.action_space)

        # self.pd, self.pi = self.pdtype.pdfromlatent(latent, init_scale=0.01)
        self.pd, self.pi = self.pdtype.dyn_pdfromlatent(latent, init_scale=0.01, index=index)

        # Take an action
        self.next_state = self.pd.sample()

        # Calculate the neg log of our probability
        self.dyn_neglogp = self.pd.neglogp(self.next_state)
        # self.dyn_neglogp = self.pd.neglogp(self.X)


        self.sess = sess or tf.get_default_session()

        # if estimate_q:
        #     assert isinstance(env.action_space, gym.spaces.Discrete)
        #     self.q = fc(vf_latent, 'dyn_q', env.action_space.n)
        #     self.vf = self.q
        # else:
        #     self.vf = fc(vf_latent, 'dyn_vf', 1)
        #     self.vf = self.vf[:,0]

    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def step(self, observation, **extra_feed):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current dynamics parameters) tuple
        """

        next_state, state, neglogp = self._evaluate([self.next_state, self.state, self.dyn_neglogp], observation, **extra_feed)
        # next_state, neglogp = self._evaluate([self.next_state,  self.dyn_neglogp], observation, **extra_feed)
        if state.size == 0:
            state = None
        return next_state, state, neglogp

    # def value(self, ob, *args, **kwargs):
    #     """
    #     Compute value estimate(s) given the observation(s)
    #
    #     Parameters:
    #     ----------
    #
    #     observation     observation data (either single or a batch)
    #
    #     **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)
    #
    #     Returns:
    #     -------
    #     value estimate
    #     """
    #     return self._evaluate(self.vf, ob, *args, **kwargs)

    def save(self, save_path):
        tf_util_ensemble.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util_ensemble.load_state(load_path, sess=self.sess)

def build_dynamics(env, dynamics_network, value_network=None, normalize_observations=False, estimate_q=False, **dynamics_kwargs):
    if isinstance(dynamics_network, str):
        network_type = dynamics_network
        dynamics_network = get_network_builder(network_type)(**dynamics_kwargs)

    def dynamics_fn(nbatch=None, nsteps=None, sess=None, observ_placeholder=None, index=None):
        ob_space = env.observation_space
        # ac_space = env.action_space
        # print("shape", (64,) + (ob_space.shape[0] + ac_space.shape[0], ))
        # Assume we have the same type for state and action space (Continuous - Continuous, Discrete - Discrete)
        # assert isinstance(ob_space, Discrete) or isinstance(ob_space, Box) or isinstance(ob_space, MultiDiscrete), \
        #     'Can only deal with Discrete and Box observation spaces for now'
        #
        # dtype = ob_space.dtype
        # if dtype == np.int8:
        #     dtype = np.uint8

        #X = tf.placeholder(shape=(nbatch,) + (ob_space.shape[0] + ac_space.shape[0], ), dtype=dtype, name='dyn_input')

        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch)


        extra_tensors = {}

        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            encoded_x = X

        encoded_x = encode_observation(ob_space, encoded_x) #  Encode input in the way that is appropriate to the observation space(float)

        with tf.variable_scope('dyn%s'%index, reuse=tf.AUTO_REUSE):
            dynamics_latent = dynamics_network(encoded_x)
            if isinstance(dynamics_latent, tuple):
                dynamics_latent, recurrent_tensors = dynamics_latent

                if recurrent_tensors is not None:
                    # recurrent architecture, need a few more steps
                    nenv = nbatch // nsteps
                    assert nenv > 0, 'Bad input for recurrent dynamics: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
                    dynamics_latent, recurrent_tensors = dynamics_network(encoded_x, nenv)
                    extra_tensors.update(recurrent_tensors)

        #             print('dynamics%s'%character, train_dynamics_model[i])

        ### original
        # with tf.variable_scope('dyn', reuse=tf.AUTO_REUSE):
        #     dynamics_latent = dynamics_network(encoded_x)
        #     if isinstance(dynamics_latent, tuple):
        #         dynamics_latent, recurrent_tensors = dynamics_latent
        #
        #         if recurrent_tensors is not None:
        #             # recurrent architecture, need a few more steps
        #             nenv = nbatch // nsteps
        #             assert nenv > 0, 'Bad input for recurrent dynamics: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
        #             dynamics_latent, recurrent_tensors = dynamics_network(encoded_x, nenv)
        #             extra_tensors.update(recurrent_tensors)

        ### original delete  tf.variable_scope (first line)
        # dynamics_latent = dynamics_network(encoded_x)
        # if isinstance(dynamics_latent, tuple):
        #     dynamics_latent, recurrent_tensors = dynamics_latent
        #
        #     if recurrent_tensors is not None:
        #         # recurrent architecture, need a few more steps
        #         nenv = nbatch // nsteps
        #         assert nenv > 0, 'Bad input for recurrent dynamics: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
        #         dynamics_latent, recurrent_tensors = dynamics_network(encoded_x, nenv)
        #         extra_tensors.update(recurrent_tensors)


        # _v_net = value_network
        #
        # if _v_net is None or _v_net == 'shared':
        #     vf_latent = dynamics_latent
        # else:
        #     if _v_net == 'copy':
        #         _v_net = dynamics_network
        #     else:
        #         assert callable(_v_net)
        #
        #     with tf.variable_scope('dyn_vf', reuse=tf.AUTO_REUSE):
        #         vf_latent = _v_net(encoded_x)

        dynamics = DynamicsWithValue(
            env=env,
            observations=X,
            latent=dynamics_latent,
            sess=sess,
            index=index, ### added
            **extra_tensors
        )
        return dynamics

    return dynamics_fn


def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms
