# -*- coding: utf-8 -*-
# @Author: shubham
# @Date:   2017-01-10 19:37:24
# @Last Modified by:   shubham
# @Last Modified time: 2017-01-16 01:16:52

import gym
from gym import wrappers
import pandas as pd
import numpy as np
import matplotlib

import sklearn.pipeline
import sklearn.preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

import sys
import random
import itertools

if "../" not in sys.path:
	sys.path.append("../")

from lib import plotting
matplotlib.style.use('ggplot')


class Estimator(object):
	def __init__(self, n, scaler, featurizer, sample_state):
		self.n = n
		self.scaler = scaler
		self.featurizer = featurizer

		self.models = []
		for _ in range(n):
			clf = SGDRegressor(learning_rate="constant")
			
			# bit of hack
			features = [self.featurize(sample_state)]
			target = [0]
			clf.partial_fit(features, target)
			
			self.models.append(clf)

	def featurize(self, state):
		scaled = self.scaler.transform([state])
		featurized = self.featurizer.transform(scaled)
		return featurized[0]

	def hack(self, state):
		[self.update(state, a, 0) for a in range(self.n)]

	def predict(self, state):
		features = [self.featurize(state)]
		return [model.predict(features)[0] for model in self.models]

	def update(self, state, action, target):
		features = [self.featurize(state)]
		target = [target]
		self.models[action].partial_fit(features, target)


class Agent(object):
	def __init__(self, nA, scaler, featurizer, sample_state, epsilon=0.5, epsilon_decay=0.99, gamma=1):
		self.nA = nA
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.gamma = gamma
		self.Q = Estimator(nA, scaler, featurizer, sample_state)
	
	def policy(self, state):
		# select greedy action
		action_selected = np.argmax(self.Q.predict(state))
		
		# assign epsilon prob to each action
		action_prob = np.ones(self.nA, dtype=float) * self.epsilon / self.nA
		action_prob[action_selected] += (1-self.epsilon)

		# select an action base on prob
		action = np.random.choice(self.nA, p=action_prob)

		return action

	def set_initial_state(self, state):
		self.state = state
		self.action = self.policy(state)
		return self.action
	
	def act(self, next_state, reward):
		state = self.state
		action = self.action
		gamma = self.gamma

		# td update
		td_target = reward + gamma * np.max(self.Q.predict(next_state))
		self.Q.update(state, action, td_target)

		self.state = next_state
		self.action = self.policy(next_state)
		return self.action

def main():
	env = gym.make('MountainCar-v0')
	outdir = './experiment-results'
	# env = wrappers.Monitor(env, directory=outdir, force=True)

	# Keeps track of useful statistics
	num_episodes = 300
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))

	# Feature Preprocessing: Normalize to zero mean and unit variance
	# We use a few samples from the observation space to do this
	observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
	scaler = sklearn.preprocessing.StandardScaler()
	scaler.fit(observation_examples)

	# Used to converte a state to a featurizes represenation.
	# We use RBF kernels with different variances to cover different parts of the space
	featurizer = sklearn.pipeline.FeatureUnion([
			("rbf1", RBFSampler(gamma=5.0, n_components=100)),
			("rbf2", RBFSampler(gamma=2.0, n_components=100)),
			("rbf3", RBFSampler(gamma=1.0, n_components=100)),
			("rbf4", RBFSampler(gamma=0.5, n_components=100))
			])
	featurizer.fit(scaler.transform(observation_examples))

	agent = Agent(env.action_space.n, scaler, featurizer, env.observation_space.sample(), epsilon=0, gamma=1)
	
	for i_episode in range(num_episodes):
		print("\rEpisode {}/{} ({})".format(i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")
		sys.stdout.flush()
		
		state = env.reset()
		action = agent.set_initial_state(state)

		for t in itertools.count():
			next_state, reward, done, info = env.step(action)
			action = agent.act(next_state, reward)
			
			# Update statistics
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			if done:
				break
	
	env.close()
	# gym.upload(outdir, api_key='sk_9YxUhFDaT5XSahcLut47w')

	plotting.plot_cost_to_go_mountain_car(env, agent.Q)
	plotting.plot_episode_stats(stats, smoothing_window=25)


if __name__ == '__main__':
	main()

