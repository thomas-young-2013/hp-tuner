import gym
import itertools
import matplotlib
import numpy as np
import sys
import tensorflow as tf
import collections

from mfes.evaluate_function.lib.cliff_walking import CliffWalkingEnv
from mfes.evaluate_function.lib import plotting
from mfes.utils.ease import ease_target
matplotlib.style.use('ggplot')
env = CliffWalkingEnv()


class PolicyEstimator():
    """
    Policy Function approximator.
    """

    def __init__(self, learning_rate=0.01, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.int32, [], "state")
            self.action = tf.placeholder(dtype=tf.int32, name="action")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # This is just table lookup estimator
            state_one_hot = tf.one_hot(self.state, int(env.observation_space.n))
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(state_one_hot, 0),
                num_outputs=env.action_space.n,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            self.action_probs = tf.squeeze(tf.nn.softmax(self.output_layer))
            self.picked_action_prob = tf.gather(self.action_probs, self.action)

            # Loss and train op
            self.loss = -tf.log(self.picked_action_prob) * self.target

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, {self.state: state})

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.target: target, self.action: action}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class ValueEstimator():
    """
    Value Function approximator.
    """

    def __init__(self, learning_rate=0.1, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.int32, [], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # This is just table lookup estimator
            state_one_hot = tf.one_hot(self.state, int(env.observation_space.n))
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(state_one_hot, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, {self.state: state})

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


def reinforce(env, estimator_policy, estimator_value, num_episodes, discount_factor=1.0, session=None):
    """
    REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient.

    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized
        estimator_value: Value function approximator, used as a baseline
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    MAX_STEPS = 150
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    for i_episode in range(num_episodes):
        # Reset the environment and pick the fisrst action
        state = env.reset()

        episode = []

        # One step in the environment
        for t in itertools.count():

            # Take a step
            action_probs = estimator_policy.predict(state, sess=session)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            # Keep track of the transition
            episode.append(Transition(
                state=state, action=action, reward=reward, next_state=next_state, done=done))

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # Print out which step we're on, useful for debugging.
            print("\rStep {} @ Episode {}/{} ({})".format(
                t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")
            # sys.stdout.flush()

            if done or t > MAX_STEPS:
                break

            state = next_state

        # Go through the episode and make policy updates
        for t, transition in enumerate(episode):
            # The return after this timestep
            total_return = sum(discount_factor ** i * t.reward for i, t in enumerate(episode[t:]))
            # Calculate baseline/advantage
            baseline_value = estimator_value.predict(transition.state, sess=session)
            advantage = total_return - baseline_value
            # Update our value estimator
            estimator_value.update(transition.state, total_return, sess=session)
            # Update our policy estimator
            estimator_policy.update(transition.state, advantage, transition.action, sess=session)
    print('='*30, 'start to validate')
    # random validation.
    num_test_episodes = 10
    test_reward = [0]*num_test_episodes
    for i_episode in range(num_test_episodes):
        # Reset the environment and pick the fisrst action
        state = env.reset()
        # One step in the environment
        for t in itertools.count():
            # Take a step
            action_probs = estimator_policy.predict(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            test_reward[i_episode] += reward
            if done or t > MAX_STEPS:
                break
            state = next_state
    return np.mean(test_reward)


@ease_target(model_dir="./data/models", name='reinforce')
def train(epoch_num, params, logger=None):
    num_episodes = int(epoch_num) * 100
    discount_factor = params['discount_factor']
    policy_learning_rate = params['policy_learning_rate']
    value_learning_rate = params['value_learning_rate']

    import tensorflow as tf
    num_repeat = 1
    reward_list = []

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    for i in range(num_repeat):
        tf.reset_default_graph()
        global_step = tf.Variable(0, name="global_step", trainable=False)
        policy_estimator = PolicyEstimator(learning_rate=policy_learning_rate)
        value_estimator = ValueEstimator(learning_rate=value_learning_rate)
        # Start training
        with tf.Session(graph=tf.get_default_graph(), config=config) as sess:
            sess.run(tf.initialize_all_variables())
            # Note, due to randomness in the policy the number of episodes you need to learn a good
            # policy may vary. ~2000-5000 seemed to work well for me.
            test_reward = reinforce(env, policy_estimator, value_estimator, num_episodes,
                                    discount_factor=discount_factor, session=sess)
        reward_list.append(test_reward)

    print('reward list', reward_list)
    loss_value = max(reward_list)*-1
    result = {'loss': loss_value}
    return result
