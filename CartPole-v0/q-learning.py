# -*- coding: utf-8 -*-
# @Author: shubham
# @Date:   2017-01-10 19:37:24
# @Last Modified by:   shubham
# @Last Modified time: 2017-01-10 23:09:02

import gym
from gym import wrappers

import pandas as pd
import numpy as np

import sys
import random
import itertools
from pprint import pprint
from collections import defaultdict

class Agent(object):
	def __init__(self, nA=4, epsilon=0.1, alpha=0.5, gamma=1):
		self.nA = nA
		self.epsilon = epsilon
		self.alpha = alpha
		self.gamma = gamma
		self.Q = defaultdict(lambda: np.random.uniform(size=nA))
	

	def policy(self, state):
		# select greedy action
		action_selected = np.argmax(self.Q[state])
		
		# assign epsilon prob to each action
		action_prob = np.full(self.nA, self.epsilon/self.nA)
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
		alpha = self.alpha
		gamma = self.gamma

		# TD Update
		td_target = reward + gamma * np.max(self.Q[next_state])
		td_error = td_target - self.Q[state][action]
		self.Q[state][action] += alpha * td_error
		
		# select next action eps-greedy
		next_action = self.policy(next_state)
		self.state = next_state
		self.action = next_action
		return self.action


def build_state(observation):
	bins_range = [(-2.4, 2.4), (-2, 2), (-1, 1), (-3.5, 3.5)]
	bins = [np.linspace(mn, mx, 10) for mn, mx in bins_range]
	
	state = ''
	for feature, _bin in zip(observation, bins):
		state += str(np.digitize(feature, _bin))
	return state

def main():
	env = gym.make('CartPole-v0')
	outdir = './experiment-results'
	# env = wrappers.Monitor(env, directory=outdir, force=True)

	agent = Agent(env.action_space.n)
	for i_episode in range(100000):
		observation = env.reset()
		state = build_state(observation)
		action = agent.set_initial_state(state)

		episode_reward = 0
		for t in itertools.count():
			next_ob, reward, done, info = env.step(action)
			episode_reward += reward
			next_state = build_state(next_ob)
			action = agent.act(next_state, reward)
			if done:
				break

		if (i_episode + 1) % 100 == 0:
			print("\rEpisode: {}, Reward: {}".format(i_episode + 1, episode_reward), end="")
			sys.stdout.flush()
	
	env.close()
	# gym.upload(outdir, api_key='sk_9YxUhFDaT5XSahcLut47w')


if __name__ == '__main__':
	main()

