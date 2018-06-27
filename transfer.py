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

class SoftmaxPolicy:
    def __init__(self, rng, nfeatures, nblocks, nactions, temp=1.):
        self.rng = rng
        self.weights = np.zeros((nfeatures, nblocks, nactions))
        self.temp = temp

    def value(self, phi, block, action=None):
        if action is None:
            return np.sum(self.weights[phi, block, :], axis=0)
        return np.sum(self.weights[phi, block, action], axis=0)

    def pmf(self, phi, block):
        v = self.value(phi, block)/self.temp
        return np.exp(v - logsumexp(v))

    def sample(self, phi, block):
        return int(self.rng.choice(self.weights.shape[2], p=self.pmf(phi, block)))

class SigmoidTermination:
    def __init__(self, rng, nfeatures, nblocks):
        self.rng = rng
        self.weights = np.zeros((nfeatures, nblocks))

    def pmf(self, phi, block):
        return expit(np.sum(self.weights[phi, block]))

    def sample(self, phi, block):
        return int(self.rng.uniform() < self.pmf(phi, block))

    def grad(self, phi, block):
        terminate = self.pmf(phi, block)
        return terminate*(1. - terminate), phi, block

class IntraOptionQLearning:
    def __init__(self, discount, lr, terminations, weights):
        self.lr = lr
        self.discount = discount
        self.terminations = terminations
        self.weights = weights

    def start(self, phi, block, option):
        self.last_phi = phi
        self.last_block = block
        self.last_option = option
        self.last_value = self.value(phi, block, option)

    def value(self, phi, block, option=None):
        if option is None:
            return np.sum(self.weights[phi, block, :], axis=0)
        return np.sum(self.weights[phi, block, option], axis=0)

    def advantage(self, phi, block, option=None):
        values = self.value(phi, block)
        advantages = values - np.max(values)
        if option is None:
            return advantages
        return advantages[option]

    def update(self, phi, block, option, reward, done):
        # One-step update target
        update_target = reward
        if not done:
            current_values = self.value(phi, block)
            termination = self.terminations[self.last_option].pmf(phi, block)
            update_target += self.discount*((1. - termination)*current_values[self.last_option] + termination*np.max(current_values))

        # Dense gradient update step
        tderror = update_target - self.last_value
        self.weights[self.last_phi, self.last_block, self.last_option] += self.lr*tderror

        if not done:
            self.last_value = current_values[option]
        self.last_option = option
        self.last_phi = phi
        self.last_block = block

        return update_target

class IntraOptionActionQLearning:
    def __init__(self, discount, lr, terminations, weights, qbigomega):
        self.lr = lr
        self.discount = discount
        self.terminations = terminations
        self.weights = weights
        self.qbigomega = qbigomega

    def value(self, phi, block, option, action):
        return np.sum(self.weights[phi, block, option, action], axis=0)

    def start(self, phi, block, option, action):
        self.last_phi = phi
        self.last_block = block
        self.last_option = option
        self.last_action = action

    def update(self, phi, block, option, action, reward, done):
        # One-step update target
        update_target = reward
        if not done:
            current_values = self.qbigomega.value(phi, block)
            termination = self.terminations[self.last_option].pmf(phi, block)
            update_target += self.discount*((1. - termination)*current_values[self.last_option] + termination*np.max(current_values))

        # Update values upon arrival if desired
        tderror = update_target - self.value(self.last_phi, self.last_block, self.last_option, self.last_action)
        self.weights[self.last_phi, self.last_block, self.last_option, self.last_action] += self.lr*tderror

        self.last_phi = phi
        self.last_block = block
        self.last_option = option
        self.last_action = action

class TerminationGradient:
    def __init__(self, terminations, critic, lr):
        self.terminations = terminations
        self.critic = critic
        self.lr = lr

    def update(self, phi, block, option):
        magnitude, dir_phi, dir_block = self.terminations[option].grad(phi, block)
        self.terminations[option].weights[dir_phi, dir_block] -= \
                self.lr*magnitude*(self.critic.advantage(phi, block, option))

class IntraOptionGradient:
    def __init__(self, option_policies, lr):
        self.lr = lr
        self.option_policies = option_policies

    def update(self, phi, block, option, action, critic):
        actions_pmf = self.option_policies[option].pmf(phi, block)
        self.option_policies[option].weights[phi, block, :] -= self.lr*critic*actions_pmf
        self.option_policies[option].weights[phi, block, action] += self.lr*critic

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
    # env = gym.make('Fourrooms-v0')
    env = Fourrooms()

    fname = '-'.join(['{}_{}'.format(param, val) for param, val in sorted(vars(args).items())])
    fname = 'optioncritic-dynamic-' + fname + '.npy'

    possible_next_goals = [68, 69, 70, 71, 72, 78, 79, 80, 81, 82, 88, 89, 90, 91, 92, 93, 99, 100, 101, 102, 103]

    history = np.zeros((args.nruns, args.nepisodes, 2))
    for run in range(args.nruns):
        features = Tabular(env.observation_space.n)
        nfeatures, nactions = len(features), env.action_space.n
        nblocks = 3

        # The intra-option policies are linear-softmax functions
        option_policies = [SoftmaxPolicy(rng, nfeatures, nblocks, nactions, args.temperature) for _ in range(args.noptions)]

        # The termination function are linear-sigmoid functions
        option_terminations = [SigmoidTermination(rng, nfeatures, nblocks) for _ in range(args.noptions)]

        # E-greedy policy over options
        #policy = EgreedyPolicy(rng, nfeatures, args.noptions, args.epsilon)
        policy = SoftmaxPolicy(rng, nfeatures, nblocks, args.noptions, args.temperature)

        # Different choices are possible for the critic. Here we learn an
        # option-value function and use the estimator for the values upon arrival
        critic = IntraOptionQLearning(args.discount, args.lr_critic, option_terminations, policy.weights)

        # Learn Qomega separately
        action_weights = np.zeros((nfeatures, nblocks, args.noptions, nactions))
        action_critic = IntraOptionActionQLearning(args.discount, args.lr_critic, option_terminations, action_weights, critic)

        # Improvement of the termination functions based on gradients
        termination_improvement= TerminationGradient(option_terminations, critic, args.lr_term)

        # Intra-option gradient improvement with critic estimator
        intraoption_improvement = IntraOptionGradient(option_policies, args.lr_intra)

        for episode in range(args.nepisodes):
            phi = features(env.reset())
            block = rng.randint(nblocks)
            block_time = rng.randint(10)
            env.set_block(block)
            option = policy.sample(phi, block)
            action = option_policies[option].sample(phi, block)
            critic.start(phi, block, option)
            action_critic.start(phi, block, option, action)

            cumreward = 0.
            duration = 1
            option_switches = 0
            avgduration = 0.
            for step in range(args.nsteps):
                observation, reward, done, _ = env.step(action)
                phi = features(observation)

                # Termination might occur upon entering the new state
                if option_terminations[option].sample(phi, block):
                    option = policy.sample(phi, block)
                    option_switches += 1
                    avgduration += (1./option_switches)*(duration - avgduration)
                    duration = 1

                action = option_policies[option].sample(phi, block)

                # Critic update
                update_target = critic.update(phi, block, option, reward, done)
                action_critic.update(phi, block, option, action, reward, done)

                if isinstance(option_policies[option], SoftmaxPolicy):
                    # Intra-option policy update
                    critic_feedback = action_critic.value(phi, block, option, action)
                    if args.baseline:
                        critic_feedback -= critic.value(phi, block, option)
                    intraoption_improvement.update(phi, block, option, action, critic_feedback)

                    # Termination update
                    termination_improvement.update(phi, block, option)

                cumreward += reward
                duration += 1
                block_time -= 1
                if block_time < 0:
                    block = rng.randint(nblocks)
                    block_time = rng.randint(10)
                    env.set_block(block)
                if done:
                    break

            history[run, episode, 0] = step
            history[run, episode, 1] = avgduration
            print('Run {} episode {} steps {} cumreward {} avg. duration {} switches {}'.format(run, episode, step, cumreward, avgduration, option_switches))
        np.save(fname, history)
        dill.dump({'intra_policies':option_policies, 'policy':policy, 'term':option_terminations}, open('oc-options.pl', 'wb'))
        print(fname)
