import os
import time
import functools
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance, set_global_seeds, zipsame
from baselines.common.policies import build_policy
from baselines.common.runners import AbstractEnvRunner
from baselines.common.tf_util_ensemble import get_session, save_variables, load_variables
from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer

from mpi4py import MPI
from baselines.common.tf_util_ensemble import initialize
from baselines.common.mpi_util import sync_from_root

from baselines.common.dynamics import build_dynamics
from baselines.ppo2.memory import Memory
import baselines.common.tf_util_ensemble as U
from baselines.common.cg import cg

class Model(object):

    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm):

        sess = tf.get_default_session()

        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, reuse=True)

        # CREATE THE PLACEHOLDERS
        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        # Cliprange
        CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

        # Defining Loss = - J is equivalent to max J
        pg_losses = -ADV * ratio

        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        # Total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        params = tf.trainable_variables('model')
        # 2. Build our trainer
        trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        _train = trainer.apply_gradients(grads_and_var)

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # Returns = R + yV(s')
            advs = returns - values # size 64

            # Normalize the advantages
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']


        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        if MPI.COMM_WORLD.Get_rank() == 0:
            initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        sync_from_root(sess, global_variables) #pylint: disable=E1101


class Dynamics(object):

    def __init__(self, *, dynamics, ob_space, ac_space, nbatch_act, nbatch_train, nsteps, K,
                cg_damping = 1e-2, max_kl=0.001, cg_iters=10):

        self.K = K
        nworkers = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()

        sess = get_session()

        dtype = ob_space.dtype
        if dtype == np.int8:
            dtype = np.uint8

        input_dyn = []
        train_dynamics_model = []
        old_dynamics_model = []

        ### Make K dynamics models
        for i in range(self.K):
            character = chr(i+65) # 1 -> A(65), ... convert to ASCII code
            input_dyn.append(tf.placeholder(shape=(None,) + (ob_space.shape[0] + ac_space.shape[0], ),
                                            dtype=dtype, name='dyn_input%s'%character))

            with tf.variable_scope('dynamics%s'%character, reuse=tf.AUTO_REUSE):
                train_dynamics_model.append( dynamics(nbatch_train, nsteps, sess, observ_placeholder=input_dyn[i], index=character) )

            with tf.variable_scope('old_dynamics%s'%character, reuse=tf.AUTO_REUSE):
                old_dynamics_model.append( dynamics(nbatch_train, nsteps, sess, observ_placeholder=input_dyn[i], index=character) )


        ####################################
        ### Codes for Dynamics Update

        ob_dyn_next = []
        klnewold_dyn = []
        meankl_dyn = []
        #
        objective = []
        old_objective = []
        MDP_model_var_list = []
        #
        MDP_get_flat = []
        MDP_set_from_flat = []
        MDP_klgrads = []
        #
        MDP_flat_tangent = []
        MDP_fvp = []
        #
        MDP_assign_old_eq_new = []
        MDP_compute_fvp = []
        #
        only_objective = []
        only_old_objective = []

        objective_array = []
        objective_array_grad = []

        MDP_compute_kl = []
        l2_norm = []

        old_MDP_model_var_list = []
        old_MDP_get_flat = []
        old_MDP_set_from_flat = []

        MDP_compute_l2_norm_grad = []


        for i in range(self.K):
            charact = chr(i+65)
            ob_dyn_next.append(train_dynamics_model[i].pdtype.sample_placeholder([None]))
            klnewold_dyn.append(train_dynamics_model[i].pd.kl(old_dynamics_model[i].pd))
            meankl_dyn.append(tf.reduce_mean(klnewold_dyn[i]))

            MDP_var_list_temp = get_trainable_variables('dynamics%s'%charact)

            MDP_model_var_list.append(MDP_var_list_temp)

            ### Old Parameters
            MDP_var_list_temp2 = get_trainable_variables('old_dynamics%s'%charact)

            old_MDP_model_var_list.append(MDP_var_list_temp2)


            objective.append( tf.reduce_mean(  train_dynamics_model[i].pd.logp(ob_dyn_next[i])  ) ) # (?, )
            old_objective.append( tf.reduce_mean( old_dynamics_model[i].pd.logp(ob_dyn_next[i] ) ) )

            MDP_var_list = MDP_model_var_list[i]
            a = U.GetFlat(MDP_var_list)
            b = U.SetFromFlat(MDP_var_list)

            MDP_get_flat.append(a)
            MDP_set_from_flat.append(b)

            ############ old set and get functions
            old_MDP_var_list = old_MDP_model_var_list[i]

            old_a = U.GetFlat(old_MDP_var_list)
            old_b = U.SetFromFlat(old_MDP_var_list)

            old_MDP_get_flat.append(old_a)
            old_MDP_set_from_flat.append(old_b)

            MDP_klgrads.append(tf.gradients(meankl_dyn[i], MDP_var_list))

            MDP_flat_tangent.append(tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan%s" % charact))

            shapes = [var.get_shape().as_list() for var in MDP_var_list]
            start = 0
            tangents = []
            for shape in shapes:
                sz = U.intprod(shape)
                tangents.append(tf.reshape(MDP_flat_tangent[i][start:start + sz], shape))
                start += sz

            l2_temp = 0
            for var in MDP_var_list:
                l2_temp += tf.reduce_sum(tf.square(var))

            l2_norm.append(l2_temp)

            gvp = tf.add_n(
                [tf.reduce_sum(g * tangent) for (g, tangent) in zipsame(MDP_klgrads[i], tangents)])  # pylint: disable=E1111
            fvp = U.flatgrad(gvp, MDP_var_list)
            MDP_fvp.append(fvp)

            MDP_assign_old_eq_new.append(U.function([], [], updates=[tf.assign(oldv, newv)
                                                                     for (oldv, newv) in
                                                                     zipsame(get_variables('old_dynamics%s'%charact),
                                                                             get_variables('dynamics%s'%charact))]))


            MDP_compute_fvp.append(U.function([MDP_flat_tangent[i], input_dyn[i], ob_dyn_next[i]], MDP_fvp[i]))
        #
            only_objective.append(U.function([input_dyn[i], ob_dyn_next[i]], objective[i]))

            only_old_objective.append(U.function([input_dyn[i], ob_dyn_next[i]], old_objective[i]))

            objective_array.append( U.function( [ input_dyn[i], ob_dyn_next[i] ], train_dynamics_model[i].pd.logp(ob_dyn_next[i]) )  )

            objective_array_grad.append(
                U.function([input_dyn[i], ob_dyn_next[i]],
                           U.flatgrad(train_dynamics_model[i].pd.logp(ob_dyn_next[i]), MDP_model_var_list[i]))) ## grad size: (6818, )

            MDP_compute_kl.append(U.function([input_dyn[i], ob_dyn_next[i]], meankl_dyn[i]))

        l2_norm_sum = 0
        for i in range(K):
            l2_norm_sum += l2_norm[i]

        MDP_compute_l2_norm_sum = U.function([], l2_norm_sum)

        for i in range(K):
            MDP_compute_l2_norm_grad.append(U.function([ ], U.flatgrad(l2_norm_sum, MDP_model_var_list[i])   ))

        def allmean(x):
            assert isinstance(x, np.ndarray)
            out = np.empty_like(x)
            MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
            out /= nworkers
            return out

        U.initialize()

        for i in range(self.K):
            phi_init = MDP_get_flat[i]()
            MPI.COMM_WORLD.Bcast(phi_init, root=0)
            MDP_set_from_flat[i](phi_init)


        def dynamics_train(index, state, action, next_state, dyn_weights, alpha):

            j = index

            size_num = np.size(state, axis=0)
            dynamics_state_and_action = np.concatenate((state, action), axis=1)

            args = dynamics_state_and_action, next_state

            fvpargs = [arr[::5] for arr in args]

            def MDP_fisher_vector_product(p):
                return allmean(MDP_compute_fvp[j](p, *fvpargs)) + cg_damping * p

            MDP_assign_old_eq_new[j]()

            surr_begin = allmean(np.array(compute_likelihood(dyn_weights, K, *args, size_num)))

            surr_begin_l2_norm = allmean(np.array(MDP_compute_l2_norm_sum()))

            surr_begin = surr_begin - alpha*surr_begin_l2_norm

            g = ensemble_compute_grad(dyn_weights, K, *args, size_num, j)

            g_l2_norm = MDP_compute_l2_norm_grad[j]()
            g = g - alpha*g_l2_norm

            g = allmean(g)
            if np.allclose(g, 0):
                logger.log("Got zero gradient. not updating")
            else:
                stepdir = cg(MDP_fisher_vector_product, g, cg_iters=cg_iters, verbose=rank == 0)  # s \simeq A^{-1}g
                assert np.isfinite(stepdir).all()
                shs = .5 * stepdir.dot(MDP_fisher_vector_product(stepdir))  # shs = 1/2 s^{T}As
                lm = np.sqrt(shs / max_kl)  # lm = 1 / \beta
                fullstep = stepdir / lm  # fullstep = \beta s, and then shrink exponentially (10 times)
                expectedimprove = g.dot(fullstep)
                surrbefore = surr_begin
                stepsize = 1.0
                phibefore = MDP_get_flat[j]()
                for _ in range(10):
                    phinew = phibefore + fullstep * stepsize
                    MDP_set_from_flat[j](phinew)
                    surr = allmean(np.array(compute_likelihood(dyn_weights, K, *args, size_num)))
                    surr_l2_norm = allmean(np.array(MDP_compute_l2_norm_sum()))
                    surr = surr - alpha * surr_l2_norm
                    kl = allmean(np.array(MDP_compute_kl[j](*args)))
                    meanlosses = surr, kl
                    improve = surr - surrbefore
                    if not np.isfinite(meanlosses).all():
                        pass
                    elif kl > max_kl * 1.5:
                        pass
                    elif improve < 0:
                        pass
                    else:
                        break
                    stepsize *= .5
                else:
                    MDP_set_from_flat[j](phibefore)
                print("Updating the model %i is completed" % (j + 1))


        def ensemble_compute_grad(new_weights, num, dynamics_state_and_action, next_state, size, model_index):

            args = dynamics_state_and_action, next_state

            list_of_grad = []

            if num == 1:
                for idx in range(size):
                    args_temp = dynamics_state_and_action[idx], next_state[idx]
                    list_of_grad.append(objective_array_grad[0](*args_temp))
            else:
                if np.isin(1.0, new_weights): ## Case for calculating directly not using EM method
                    index_of_one = np.argmax(new_weights)
                    for idx in range(size):
                        args_temp = dynamics_state_and_action[idx], next_state[idx]
                        list_of_grad.append(objective_array_grad[index_of_one](*args_temp))
                else:
                    log_probability_list = []
                    for idx in range(num):
                        log_probability_list.append(objective_array[idx](*args))

                    mean_log_probability = np.max(log_probability_list, axis=0)

                    for idx in range(num):
                        log_probability_list[idx] -= mean_log_probability

                    numer_list = []
                    for idx in range(num):
                        value_temp = np.exp(log_probability_list[idx], dtype=np.longfloat)
                        numer_list.append(new_weights[idx] * value_temp)

                    denom = np.zeros(size, dtype=np.longfloat)

                    for idx in range(num):
                        denom += numer_list[idx]

                    for idx in range(size):
                        if denom[idx] > 0.0:
                            args_temp = dynamics_state_and_action[idx], next_state[idx]
                            batch_of_gradients = objective_array_grad[model_index](*args_temp) ## np.float32
                            list_of_grad.append(numer_list[model_index][idx] / denom[idx] * batch_of_gradients)
                        else:
                            print("denom[%s] == 0.0 for compute_grad" % idx)

            new_gradients = np.mean(list_of_grad, axis=0)

            return new_gradients


        def compute_likelihood(new_weights, num, dynamics_state_and_action, next_state, size):

            args = dynamics_state_and_action, next_state

            if num == 1:
                log_likelihood_array = objective_array[0](*args)
            else:
                if np.isin(1.0, new_weights): ## Case for calculating directly not using EM method
                    index_of_one = np.argmax(new_weights)
                    log_likelihood_array = objective_array[index_of_one](*args)
                else:
                    log_probability_list = []
                    for idx in range(num):
                        log_probability_list.append(objective_array[idx](*args))

                    mean_log_probability = np.max(log_probability_list, axis=0)

                    for idx in range(num):
                        log_probability_list[idx] -= mean_log_probability

                    numer_list = []
                    for idx in range(num):
                        value_temp = np.exp(log_probability_list[idx], dtype=np.longfloat)
                        numer_list.append(new_weights[idx] * value_temp)

                    denom = np.zeros(size, dtype=np.longfloat)

                    for idx in range(num):
                        denom += numer_list[idx]

                    denom2 = []
                    mean_log_probability2 = []

                    for idx in range(size):
                        if denom[idx] > 0.0:
                            denom2.append(denom[idx])
                            mean_log_probability2.append(mean_log_probability[idx])
                        else:
                            print("denom[%s] == 0.0 for compute_likelihood" % idx)

                    log_likelihood_array = np.log(denom2) + mean_log_probability2

            likelihood = np.mean(log_likelihood_array)

            return likelihood


        ### Used only when K >= 2
        def weight_train(old_weight, num, state, action, next_state, size, num_iter, total_iter, exponent):

            dynamics_state_and_action = np.concatenate((state, action), axis=1)

            args = dynamics_state_and_action, next_state
            if np.isin(1.0, old_weight): ## Case for calculating directly not using EM method
                new_weights = np.zeros([num], dtype=np.longfloat)
                index_of_one = np.argmax(old_weight)
                new_weights[index_of_one] = 1.0
            else:
                log_probability_list = []
                for idx in range(num):
                    log_probability_list.append(objective_array[idx](*args))

                mean_log_probability = np.max(log_probability_list, axis=0)


                for idx in range(num):
                    log_probability_list[idx] -= mean_log_probability

                numer_list = []
                list_of_list = []
                for idx in range(num):
                    value_temp = np.exp(log_probability_list[idx], dtype=np.longfloat)
                    numer_list.append(old_weight[idx] * value_temp)
                    numer_list[idx] = numer_list[idx] ** exponent
                    if idx < num - 1:
                        list_of_list.append([])

                denom = np.zeros(size, dtype=np.longfloat)

                for idx in range(num):
                    denom += numer_list[idx]

                for idx in range(size):
                    if denom[idx] > 0.0:
                        for iterate in range(num - 1):
                            list_of_list[iterate].append(numer_list[iterate][idx] / denom[idx])
                    else:
                        print("denom[%s] == 0.0 for weight_train" % idx)

                new_weights = []

                for idx in range(num - 1):
                    new_weights.append(np.mean(list_of_list[idx], dtype=np.longfloat))

                temp_ones = np.ones(1, dtype=np.longfloat)
                new_weights.append(temp_ones[0] - np.sum(new_weights, dtype=np.longfloat))

            return new_weights



        ### train dynamics models
        self.dynamics_train = dynamics_train
        self.weight_train = weight_train

        self.only_objective = only_objective
        self.only_old_objective = only_old_objective

        self.compute_likelihood = compute_likelihood

        self.MDP_get_flat = MDP_get_flat
        self.MDP_set_from_flat = MDP_set_from_flat

        self.old_MDP_get_flat = old_MDP_get_flat
        self.old_MDP_set_from_flat = old_MDP_set_from_flat


############
class Runner(AbstractEnvRunner):

    def __init__(self, *, env, model, nsteps, gamma, lam, dynamics, index_type, ex_coef, beta, reward_freq):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma

        self.replay_memory = Memory(limit=int(1.1e6), action_shape=env.action_space.shape,
                                    observation_shape=env.observation_space.shape)

        self.dynamics = dynamics

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.index_type = index_type
        self.beta = beta

        self.ex_coef = ex_coef

        self.flag = True

        self.weights = np.ones([self.dynamics.K], dtype=np.longfloat)/self.dynamics.K
        self.old_weights = np.ones([self.dynamics.K], dtype=np.longfloat)/self.dynamics.K

        nenv = env.num_envs
        self.delay_reward = np.zeros([nenv])
        self.delay_step = np.zeros([nenv])
        self.reward_freq = reward_freq


    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_states = self.states
        epinfos = []

        int_temp = np.zeros(self.dynamics.K, dtype=np.longfloat)
        int_temp2 = np.zeros(self.dynamics.K, dtype=np.longfloat)

        selected = 0

        rews_int_temp = np.zeros(self.nsteps, 'float32')

        # For n in range number of steps
        for i in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            real_next_state_temp, rewards, self.dones, infos = self.env.step(actions)

            ### Delay Reward Part
            self.delay_reward += rewards
            self.delay_step += 1
            for n, done in enumerate(self.dones):
                if done or self.delay_step[n] == self.reward_freq:
                    rewards[n] = self.delay_reward[n]
                    self.delay_reward[n] = self.delay_step[n] = 0
                else:
                    rewards[n] = 0
            mb_rewards.append(rewards)

            B = self.obs.shape[0]
            for b in range(B):
                self.replay_memory.append(self.obs[b], actions[b], real_next_state_temp[b])

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)


            ###### Generating Intrinsic Reward (Without Normalizing Intrinsic Reward)
            ######
            if self.dynamics.K != 0:
                state_and_action = np.concatenate((self.obs, actions), axis=1)
                state_and_action_s = np.reshape(state_and_action, [-1, self.state_dim + self.action_dim])
                ob_next_s = np.reshape(real_next_state_temp, [-1, self.state_dim])
                args = state_and_action_s, ob_next_s

                if self.dynamics.K == 1:  ### Single Surprise
                    rews_int_temp[i] = self.dynamics.only_objective[0](*args) - self.dynamics.only_old_objective[0](*args)
                else: ### K = 2, 3, 4, ...
                    if np.isin(1.0, self.weights):
                        weight_temp = self.weights
                        index_of_one = np.argmax(weight_temp)
                        if self.index_type == 'ens':  # ens
                            if np.isin(1.0, self.old_weights):
                                old_weight_temp = self.old_weights
                                old_index_of_one = np.argmax(old_weight_temp)  ###
                                rews_int_temp[i] = self.dynamics.only_objective[index_of_one](*args) \
                                                   - self.dynamics.only_old_objective[old_index_of_one](*args)
                            else:
                                old_log_probability_list = []
                                for idx2 in range(self.dynamics.K):
                                    old_log_probability_list.append(self.dynamics.only_old_objective[idx2](*args))

                                old_mean_log_probability = np.max(old_log_probability_list, axis=0)

                                for idx2 in range(self.dynamics.K):
                                    old_log_probability_list[idx2] -= old_mean_log_probability

                                for ct in range(self.dynamics.K):
                                    int_temp2[ct] = np.exp(old_log_probability_list[ct], dtype=np.longfloat)
                                denom = np.inner(self.old_weights, int_temp2)
                                if denom <= 0.0:
                                    rews_int_temp[i] = 0
                                    print(
                                        "rews_int_temp[%s] == 0.0 for 1st denominator intrinsic_reward for ens method" % i)
                                else:
                                    rews_int_temp[i] = self.dynamics.only_objective[index_of_one](*args) - np.log(
                                        denom) - old_mean_log_probability

                        elif self.index_type == 'avg':
                            log_array = []
                            for log_idx in range(self.dynamics.K):
                                log_array.append(- self.dynamics.only_old_objective[log_idx](*args))
                            log_mean = np.mean(log_array)
                            rews_int_temp[i] = self.dynamics.only_objective[index_of_one](*args) + log_mean

                        else: ### max or min
                            log_array = []
                            for log_idx in range(self.dynamics.K):
                                log_array.append(- self.dynamics.only_old_objective[log_idx](*args))
                            if self.index_type == 'max':  # max
                                selected = np.argmax(log_array)
                            elif self.index_type == 'min':  # min
                                selected = np.argmin(log_array)
                            rews_int_temp[i] = self.dynamics.only_objective[index_of_one](*args) + log_array[selected]

                    else:
                        log_probability_list = []
                        for idx in range(self.dynamics.K):
                            log_probability_list.append(self.dynamics.only_objective[idx](*args))

                        mean_log_probability = np.max(log_probability_list, axis=0)

                        for idx in range(self.dynamics.K):
                            log_probability_list[idx] -= mean_log_probability

                        for ct in range(self.dynamics.K):
                            int_temp[ct] = np.exp(log_probability_list[ct], dtype=np.longfloat)
                        numer = np.inner(self.weights, int_temp)
                        if numer <= 0.0:
                            rews_int_temp[i] = 0
                            print("rews_int_temp[%s] == 0.0 for numerator intrinsic_reward" % i)
                        else:
                            if self.flag: ### random selection
                                rand_index = np.random.randint(self.dynamics.K)
                                rews_int_temp[i] = np.log(numer) + mean_log_probability - self.dynamics.only_old_objective[rand_index](*args)
                            else:
                                if self.index_type == 'ens':
                                    if np.isin(1.0, self.old_weights):
                                        old_weight_temp = self.old_weights
                                        old_index_of_one = np.argmax(old_weight_temp)
                                        rews_int_temp[i] = np.log(numer) + mean_log_probability \
                                                           - self.dynamics.only_old_objective[old_index_of_one](*args)
                                    else:
                                        old_log_probability_list = []
                                        for idx2 in range(self.dynamics.K):
                                            old_log_probability_list.append(self.dynamics.only_old_objective[idx2](*args))

                                        old_mean_log_probability = np.max(old_log_probability_list, axis=0)

                                        for idx2 in range(self.dynamics.K):
                                            old_log_probability_list[idx2] -= old_mean_log_probability

                                        for ct in range(self.dynamics.K):
                                            int_temp2[ct] = np.exp(old_log_probability_list[ct], dtype=np.longfloat)
                                        denom = np.inner(self.old_weights, int_temp2)
                                        if denom <= 0.0:
                                            rews_int_temp[i] = 0
                                            print("rews_int_temp[%s] == 0.0 for 2nd denominator intrinsic_reward for ens method" % i)
                                        else:
                                            rews_int_temp[i] = np.log(numer) + mean_log_probability \
                                                               - np.log(denom) - old_mean_log_probability

                                elif self.index_type == 'avg':
                                    log_array = []
                                    for log_idx in range(self.dynamics.K):
                                        log_array.append(- self.dynamics.only_old_objective[log_idx](*args))
                                    log_mean = np.mean(log_array)
                                    rews_int_temp[i] = np.log(numer) + mean_log_probability + log_mean

                                else: ### max or min
                                    log_array = []
                                    for log_idx in range(self.dynamics.K):
                                        log_array.append(- self.dynamics.only_old_objective[log_idx](*args))
                                    if self.index_type == 'max':  # max
                                        selected = np.argmax(log_array)
                                    elif self.index_type == 'min':  # min
                                        selected = np.argmin(log_array)
                                    rews_int_temp[i] = np.log(numer) + mean_log_probability + log_array[selected]

            ###### Intrinsic reward normalization at the end of the batch
                if (i + 1) == self.nsteps:
                    int_mean = max(1, abs(np.mean(rews_int_temp)))
                    rews_int_temp = rews_int_temp / int_mean  # normalize
                    for index2 in range(self.nsteps):
                        mb_rewards[index2][0] = self.ex_coef*mb_rewards[index2][0] + self.beta * rews_int_temp[index2]
            ###### Intrinsic reward generate end
            #######

            self.obs[:] = real_next_state_temp


        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
                mb_states, epinfos)

    ###### We run with initial policy at the beginning of training to fill our replay buffer
    def run_begin(self):
        for _ in range(self.nsteps):
            real_actions, _, self.states, _ = self.model.step(self.obs, S=self.states, M=self.dones)

            real_next_state_temp, _, self.dones, _ = self.env.step(real_actions)

            B = self.obs.shape[0]
            for b in range(B):
                self.replay_memory.append(self.obs[b], real_actions[b], real_next_state_temp[b])

            self.obs[:] = real_next_state_temp

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def constfn(val):
    def f(_):
        return val
    return f


def learn(*, network, env, total_timesteps, eval_env = None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, load_path=None,
          begin_iter=40, model_train=4, random_initial_ratio=0, exponent=1.0,
          K=0, beta=1, alpha=0.1, index_type='min', reward_freq=40, r_ex_coef=1,
          **network_kwargs):


    ''' # K=0: Original PPO; K=1: Single Surprise # index_type= 'min', 'max', 'ens', 'avg'
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

    Parameters:
    ----------

    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                      specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                      tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                      neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                      See common/models.py/lstm for more details on using recurrent nets in policies

    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.


    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)

    total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

    ent_coef: float                   policy entropy coefficient in the optimization objective

    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                      training and 0 is the end of the training.

    vf_coef: float                    value function loss coefficient in the optimization objective

    max_grad_norm: float or None      gradient norm clipping coefficient

    gamma: float                      discounting factor

    lam: float                        advantage estimation discounting factor (lambda in the paper)

    log_interval: int                 number of timesteps between logging events

    nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.

    noptepochs: int                   number of training epochs per update

    cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                      and 0 is the end of the training

    save_interval: int                number of timesteps between saving events

    load_path: str                    path to load the model from

    **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.



    '''

    set_global_seeds(seed)

    np.set_printoptions(precision=3)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    policy = network

    dynamics = build_dynamics(env, 'mlp', **network_kwargs)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    # Instantiate the model object (that creates act_model and train_model)
    make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm)
    model = make_model()
    if load_path is not None:
        model.load(load_path)

    # Instantiate the dynamics class object
    dynamics_class = Dynamics(dynamics=dynamics, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps,K=K)

    # Instantiate the runner object
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam, dynamics=dynamics_class,
                    index_type=index_type, ex_coef=r_ex_coef, beta=beta, reward_freq=reward_freq)
    if eval_env is not None:
        eval_runner = Runner(env = eval_env, model = model, nsteps = nsteps, gamma = gamma, lam= lam, dynamics=dynamics_class,
                             index_type=index_type, ex_coef=r_ex_coef, beta=beta, reward_freq=reward_freq)


    epinfobuf = deque(maxlen=100)
    if eval_env is not None:
        eval_epinfobuf = deque(maxlen=100)

    for _ in range(begin_iter):
        runner.run_begin()


    # Start total timer
    tfirststart = time.time()

    nupdates = total_timesteps//nbatch

    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0

        logger.log("********** Iteration %i ************" % update)

        ### random_initial_ratio = 0 -> no random selection
        ### 0 <= random_initial_ratio < 1
        if update >= int(random_initial_ratio*nupdates):
            runner.flag = False


        ### Store old parameters of every model
        runner.old_weights = runner.weights

        ### Store old parameters of every model; and store new value to old one; For later calculation
        model_old_weights_list = []
        for index in range(K):
            param_temp = dynamics_class.MDP_get_flat[index]()
            model_old_weights_list.append(param_temp)

        ### Store new parameters of every model
        model_new_weights_list = []

        ### Store training sets for test
        #dyn_batch_list = []

        for index in range(K): ## We do not have to ramdomly order
            dyn_batch = runner.replay_memory.sample(batch_size=nbatch) ## Every model should train with different batches
            #dyn_batch_list.append(dyn_batch)
            dynamics_inds = np.arange(nbatch)
            for _ in range(model_train):
                # Randomize the indexes
                np.random.shuffle(dynamics_inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, nbatch_train): ## nabtch_train = 64
                    end = start + nbatch_train
                    dynamics_mbinds = dynamics_inds[start:end]
                    dynamics_slices = (arrr[dynamics_mbinds] for arrr in (dyn_batch['obs0'], dyn_batch['actions'], dyn_batch['obs1']))
                    dynamics_class.dynamics_train(index, *dynamics_slices, runner.weights, alpha)
            Param_Temp = dynamics_class.MDP_get_flat[index]()
            model_new_weights_list.append(Param_Temp)
            dynamics_class.MDP_set_from_flat[index](model_old_weights_list[index])


        if K >= 2:  # K = 2, 3, 4, ...
            q_batch = runner.replay_memory.sample(batch_size=nbatch)
            dynamics_inds = np.arange(nbatch)
            for _ in range(model_train):
                # Randomize the indexes
                np.random.shuffle(dynamics_inds)
                # 0 to batch_size with batch_train_size step
                for start2 in range(0, nbatch, nbatch_train):  ## nabtch_train = 64
                    end2 = start2 + nbatch_train
                    dynamics_mbinds = dynamics_inds[start2:end2]
                    q_slices = (arrry[dynamics_mbinds] for arrry in
                                (q_batch['obs0'], q_batch['actions'], q_batch['obs1']))
                    runner.weights = dynamics_class.weight_train(runner.weights, K, *q_slices, nbatch_train,
                                                                 update, nupdates, exponent)

        ### Setting new parameters simultaneously
        for index in range(K):
            dynamics_class.MDP_set_from_flat[index](model_new_weights_list[index])
            dynamics_class.old_MDP_set_from_flat[index](model_old_weights_list[index]) ### To make actual 1-step surprise

        # Start timer
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        # Calculate the cliprange
        cliprangenow = cliprange(frac)
        # Get minibatch
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
        if eval_env is not None:
            eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_states, eval_epinfos = eval_runner.run() #pylint: disable=E0632

        epinfobuf.extend(epinfos)
        if eval_env is not None:
            eval_epinfobuf.extend(eval_epinfos)

        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mblossvals = []
        if states is None: # nonrecurrent version
            # Index of each element of batch_size
            # Create the indices array
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    # print("slices", slices)
                    # print("*slices", *slices)
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        else: # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        # End timer
        tnow = time.time()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            #print("eprewmean", safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            if eval_env is not None:
                logger.logkv('eval_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]) )
                logger.logkv('eval_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]) )
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            if MPI.COMM_WORLD.Get_rank() == 0:
                logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir() and MPI.COMM_WORLD.Get_rank() == 0:
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)

    return model

# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def get_variables(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

def get_trainable_variables(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
