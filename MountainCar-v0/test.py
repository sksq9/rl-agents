# -*- coding: utf-8 -*-
# @Author: shubham
# @Date:   2017-01-10 19:37:24
# @Last Modified by:   shubham
# @Last Modified time: 2017-01-14 21:56:39

import gym
from gym import wrappers

import pandas as pd
import numpy as np

import random
import itertools

class Agent(object):
	def __init__(self, nA=None):
		self.nA = nA
	
	def set_initial_state(self, state):
		return np.random.choice(self.nA)
	
	def act(self, state, reward):
		return np.random.choice(self.nA)
		

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

