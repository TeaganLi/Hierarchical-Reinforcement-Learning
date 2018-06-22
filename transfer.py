import gym
import argparse
import numpy as np
from fourrooms import Fourrooms

from scipy.special import expit
from scipy.misc import logsumexp
import dill

class Tabular:
    def __init__(self, nstates):
        self.nstates = nstates

    def __call__(self, state):
        return np.array([state,])

    def __len__(self):
        return self.nstates

class EgreedyPolicy:
    def __init__(self, rng, nfeatures, nactions, epsilon):
        self.rng = rng
        self.epsilon = epsilon
        self.weights = np.zeros((nfeatures, nactions))

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def sample(self, phi):
        if self.rng.uniform() < self.epsilon:
            return int(self.rng.randint(self.weights.shape[1]))
        return int(np.argmax(self.value(phi)))

class SoftmaxPolicy:
    def __init__(self, rng, nfeatures, nactions, temp=1.):
        self.rng = rng
        self.weights = np.zeros((nfeatures, nactions))
        self.temp = temp

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def pmf(self, phi):
        v = self.value(phi)/self.temp
        return np.exp(v - logsumexp(v))

    def sample(self, phi):
        return int(self.rng.choice(self.weights.shape[1], p=self.pmf(phi)))

class ValueFunctionLearning:
    def __init__(self, discount, lr, nfeatures):
        self.discount = discount
        self.lr = lr
        self.weights = np.zeros((nfeatures,))

    def value(self, phi):
        return np.sum(self.weights[phi], axis=0)

    def start(self, phi):
        self.last_phi = phi

    def update(self, phi, reward, done):
        # One-step update target
        update_target = reward
        if not done:
            current_value = self.value(phi)
            update_target += self.discount*current_value

        # Update values
        tderror = update_target - self.value(self.last_phi)
        self.weights[self.last_phi] += self.lr*tderror

        self.last_phi = phi
        return tderror

class PolicyGradient:
    def __init__(self, policy, lr):
        self.lr = lr
        self.policy = policy

    def update(self, phi, action, critic):
        actions_pmf = self.policy.pmf(phi)
        self.policy.weights[phi, :] -= self.lr*critic*actions_pmf
        self.policy.weights[phi, action] += self.lr*critic


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--discount', help='Discount factor', type=float, default=0.99)
    parser.add_argument('--lr_term', help="Termination gradient learning rate", type=float, default=1e-3)
    parser.add_argument('--lr_intra', help="Intra-option gradient learning rate", type=float, default=1e-3)
    parser.add_argument('--lr_critic', help="Learning rate", type=float, default=1e-2)
    parser.add_argument('--epsilon', help="Epsilon-greedy for policy over options", type=float, default=1e-2)
    parser.add_argument('--nepisodes', help="Number of episodes per run", type=int, default=250)
    parser.add_argument('--nruns', help="Number of runs", type=int, default=100)
    parser.add_argument('--nsteps', help="Maximum number of steps per episode", type=int, default=1000)
    parser.add_argument('--noptions', help='Number of options', type=int, default=4)
    parser.add_argument('--baseline', help="Use the baseline for the intra-option gradient", action='store_true', default=False)
    parser.add_argument('--temperature', help="Temperature parameter for softmax", type=float, default=1e-2)
    parser.add_argument('--primitive', help="Augment with primitive", default=False, action='store_true')

    args = parser.parse_args()

    rng = np.random.RandomState(1234)
    env = gym.make('Fourrooms-v0')

    fname = '-'.join(['{}_{}'.format(param, val) for param, val in sorted(vars(args).items())])
    fname = 'optioncritic-fourrooms-' + fname + '.npy'

    possible_next_goals = [68, 69, 70, 71, 72, 78, 79, 80, 81, 82, 88, 89, 90, 91, 92, 93, 99, 100, 101, 102, 103]

    history = np.zeros((args.nruns, args.nepisodes, 2))
    for run in range(args.nruns):
        features = Tabular(env.observation_space.n)
        nfeatures, nactions = len(features), env.action_space.n

        # The policies are linear-softmax functions
        policy = SoftmaxPolicy(rng, nfeatures, nactions, args.temperature)

        # Value function as a critic
        critic = ValueFunctionLearning(args.discount, args.lr_critic, nfeatures)

        # Policy gradient improvement with critic estimator
        policy_improvement = PolicyGradient(policy, args.lr_intra)

        for episode in range(args.nepisodes):

            phi = features(env.reset())
            action = policy.sample(phi)
            critic.start(phi)

            cumreward = 0.
            for step in range(args.nsteps):
                observation, reward, done, _ = env.step(action)
                phi = features(observation)

                # Critic update
                tderror = critic.update(phi, reward, done)

                if isinstance(policy, SoftmaxPolicy):
                    # Intra-option policy update
                    critic_feedback = tderror
                    policy_improvement.update(phi, action, tderror)

                action = policy.sample(phi)
                cumreward += reward
                if done:
                    break

            history[run, episode, 0] = step
            history[run, episode, 1] = 0
            print('Run {} episode {} steps {} cumreward {}'.format(run, episode, step, cumreward))
        np.save(fname, history)
        dill.dump({'policy':policy}, open('policy-gradient.pl', 'wb'))
        print(fname)
