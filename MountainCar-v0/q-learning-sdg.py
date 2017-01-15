# -*- coding: utf-8 -*-
# @Author: shubham
# @Date:   2017-01-10 19:37:24
# @Last Modified by:   shubham
# @Last Modified time: 2017-01-15 23:45:17

import gym
from gym import wrappers
import pandas as pd
import numpy as np

import sklearn.pipeline
import sklearn.preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

import random
import itertools

class Estimator(object):
	def __init__(self, n):
		self.n = n
		self.models = []
		for _ in range(n):
			clf = SGDRegressor()
			
			# bit of hack
			# state = env.observation_space.sample()
			# features = [self.featurizer(state)]
			# target = [0]
			# clf.partial_fit(features, target)
			
			self.models.append(clf)

	def featurizer(self, state):
		return state

	def hack(self, state):
		[self.update(state, a, 0) for a in range(self.n)]

	def predict(self, state):
		features = [self.featurizer(state)]
		return [model.predict(features)[0] for model in self.models]

	def update(self, state, action, target):
		features = [self.featurizer(state)]
		target = [target]
		self.models[action].partial_fit(features, target)


class Agent(object):
	def __init__(self, nA=4, epsilon=0.5, epsilon_decay=0.99, alpha=0.2, gamma=1):
		self.nA = nA
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.alpha = alpha
		self.gamma = gamma
		self.Q = Estimator(nA)
	
	def policy(self, state):
		# select greedy action
		action_selected = np.argmax(self.Q.predict(state))
		
		# assign epsilon prob to each action
		action_prob = np.full(self.nA, self.epsilon/self.nA)
		action_prob[action_selected] += (1-self.epsilon)

		# select an action base on prob
		action = np.random.choice(self.nA, p=action_prob)

		return action

	def set_initial_state(self, state):
		self.Q.hack(state)
		
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
	env = wrappers.Monitor(env, directory=outdir, force=True)

	agent = Agent(env.action_space.n)
	for i_episode in range(100):
		state = env.reset()
		action = agent.set_initial_state(state)
		
		for t in itertools.count():
			next_state, reward, done, info = env.step(action)
			action = agent.act(next_state, reward)
			
			if done:
				break
	
	env.close()
	# gym.upload(outdir, api_key='sk_9YxUhFDaT5XSahcLut47w')


if __name__ == '__main__':
	main()

