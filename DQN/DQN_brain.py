import tensorflow as tf
import numpy as np
import pandas as pd


# tensorboard --logdir=logs
# 0.0.0.0.6006

np.random.seed(1)
tf.set_random_seed(1)


class DeepQNetwork:
    def __int__(self,
                n_actions,      # amount of actions
                n_features,     # amount of features for observation
                learning_rate=0.01,
                reward_decay=0.9,
                e_greedy=0.9,
                replace_target_iter=300,
                memory_size=500,
                batch_size=32,
                e_greedy_increment=None,
                output_graph=False
                ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size  # capacity of experience replay
        self.batch_size = batch_size    # stochastic gradient descent
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = pd.DataFrame(np.zeros((self.memory_size, n_features*2+2)))

        # build networks of target and eval
        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter('logs/', self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []


    def _build_net(self):
        # ------------------------------ eval neural network ----------------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input state
        self.q_target = tf.placeholder(tf.float32, [None, n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10 \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)   # config of layers

            # first layer, collections is used later when assign to eval net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer, collections is used later when assign to eval net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b1', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matlul(l1, w2) + b2

        with tf.name_scope('loss'):
            self.loss = tf.reduce_sum(tf.squared_difference(self.q_target, self.q_eval))
        with tf.name_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------------------ target neural network ----------------------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input state
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10 \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)   # config of layers

            # first layer, collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer, collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b1', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matlul(l1, w2) + b2


    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):     # check if self has the attribute of memory_counter
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))     # horizontal stack to one array

        # replace old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory.iloc[index, :] = transition

        self.memory_counter += 1


    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action


    def _replace_target_params(self):
        t_params = tf.get_collection('target_net_params')   # list
        e_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])


    def learn(self):
        # check to replace target parameters



























