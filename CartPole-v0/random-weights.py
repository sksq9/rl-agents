# -*- coding: utf-8 -*-
# @Author: shubham
# @Date:   2016-12-25 23:40:18
# @Last Modified by:   shubham
# @Last Modified time: 2017-01-07 15:30:41

import gym
from gym import wrappers
import numpy as np

class Agent:
	def __init__(self, action_space, observation_space):
		self.action_space = action_space
		self.observation_space = observation_space
		self.reward = float('-inf')
		self.weights = None
		self.sloved = False

	def act(self, ob, reward):
		if not self.sloved:
			if self.weights is None:
				self.weights = np.random.rand(self.observation_space.shape[0])
		
		action = 0 if np.matmul(self.weights, ob) < 0 else 1
		return action

	def terminal(self, reward):
		if self.reward < reward:
			self.reward = reward
			self.bestWeights = self.weights
		self.weights = None
		if self.reward >= 200:
			self.sloved = True
			self.weights = self.bestWeights

def episode(env, agent):
	ob = env.reset()
	reward = 0
	totalReward = 0

	for _ in range(200):
		action = agent.act(ob, reward)
		ob, reward, done, info = env.step(action)
		totalReward += reward
		# print(action)
		
		env.render()
		if done:
			break

	agent.terminal(totalReward)
	return totalReward

def main():
	env = gym.make("CartPole-v0")
	outdir = '/tmp/random-agent-results'
	env = wrappers.Monitor(env, directory=outdir, force=True)
	
	agent = Agent(env.action_space, env.observation_space)
	num_episodes = 110
	
	for idx in range(num_episodes):
		totalReward = episode(env, agent)
		print(totalReward)
		
	env.close()
	gym.upload(outdir, api_key='sk_9YxUhFDaT5XSahcLut47w')


if __name__ == '__main__':
	main()

