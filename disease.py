from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import SingleGrid
import numpy as np
import matplotlib.pyplot as plt

class DiseaseModel(Model):
	"""
	A model with some number of agents.
	highS: Number of agents with high sociability.
	middleS: Number of agents with middle sociability.
	lowS: Number of agents with low sociability.
	width: Width of the grid.
	height: Height of the grid.
	"""
	def __init__(self, highS, middleS, lowS, width, height,rooms):
		self.num_agents = highS + middleS + lowS
		self.rooms = rooms
		if self.num_agents > width * height:
			raise ValueError("Number of agents exceeds grid capacity.")

		self.grid = SingleGrid(width, height, True)
		self.schedule = RandomActivation(self)

		# Create agents
		self.addAgents(lowS, 0, 0)
		self.addAgents(middleS, lowS, 1)
		self.addAgents(highS, lowS + highS, 2)

	def addAgents(self, n, startID, sociability):
		for i in range(n):
			a = DiseaseAgent(i + startID, sociability, self)
			self.schedule.add(a)
			# Add the agent to a random grid cell
			location = self.grid.find_empty()
			self.grid.place_agent(a, location)

	# Continue one step in simulation
	def step(self):
		self.schedule.step()

class DiseaseAgent(Agent):
	""" An agent with fixed initial disease."""
	def __init__(self, unique_id, sociability, model):
		super().__init__(unique_id, model)
		# Randomly set agent as healthy or sick
		self.disease = self.random.randrange(2)
		self.diseaserate = 1
		self.sociability = sociability
		self.resistent = []
		self.cureProb = 0.1
		self.sickTime = 0
		self.goal = self.model.rooms[self.random.randrange(len(self.model.rooms))]

	def move(self):
		""" Moves agent one step on the grid."""
		cellmates = self.model.grid.get_neighbors(self.pos,moore=True)
		if self.sociability == 0:
			if len(cellmates) > 0:
				other = self.random.choice(cellmates)
				test = ((self.pos[0] - other.pos[0]),(self.pos[1] - other.pos[1]))
				choice = (test[0]+self.pos[0],test[1]+self.pos[1])
				if self.model.grid.width > choice[0] > 0  and self.model.grid.height > choice[1] > 0:
					if model.grid.is_cell_empty(choice):
						self.model.grid.move_agent(self, choice)
						return
		if self.sociability == 1:
			for neighbour in cellmates:
				if neighbour.sociability == 2:
					return
		if self.sociability == 2:
			if len(cellmates) > 0:
				return

		x_distance = self.goal[0] - self.pos[0]
		y_distance = self.goal[1] - self.pos[1]
		if abs(x_distance) >= abs(y_distance):
			if x_distance > 0:
				choice = (self.pos[0]+1,self.pos[1])
				if 0 < choice[0] < model.grid.width and  0 < choice[1] < model.grid.height and model.grid.is_cell_empty(choice):
					self.model.grid.move_agent(self,choice)
			else:
				choice = (self.pos[0]-1,self.pos[1])
				if 0 < choice[0] < model.grid.width and  0 < choice[1] < model.grid.height and model.grid.is_cell_empty(choice):
					self.model.grid.move_agent(self,choice)
		else:
			if y_distance > 0:
				choice = (self.pos[0],self.pos[1]+1)
				if 0 < choice[0] < model.grid.width and  0 < choice[1] < model.grid.height and model.grid.is_cell_empty(choice):
					self.model.grid.move_agent(self,choice)
			else:
				choice = (self.pos[0],self.pos[1]-1)
				if 0 < choice[0] < model.grid.width and  0 < choice[1] < model.grid.height and model.grid.is_cell_empty(choice):
					self.model.grid.move_agent(self,choice)
		# possible_steps = self.model.grid.get_neighborhood(
		# 	self.pos,
		# 	moore=False,
		# 	include_center=True)
		# self.random.shuffle(possible_steps)
		# for choice in possible_steps:
		# 	if (self.goal[0] - self.pos[0] > self.goal[0] - choice[0]) or (self.goal[1] - self.pos[1] > self.goal[1] - choice[1]):
		# 		if model.grid.is_cell_empty(choice):
		# 			self.model.grid.move_agent(self, choice)

	def spread_disease(self):
		"""Spreads disease to neighbors."""
		cellmates = self.model.grid.get_neighbors(self.pos,moore=True)
		if len(cellmates) > 0:
			for i in range(len(cellmates)):
				other = cellmates[i]
				if self.disease not in other.resistent:
					if self.diseaserate > self.random.random():
						other.disease = self.disease
	def mutate(self):
		"""mutates disease in an agent."""
		if self.disease > 0:
			if 0.1 > self.random.random():
				self.disease += 1

	def cured(self):
		"""Cure agents based on cure probability sick time."""
		if self.sickTime > 6:
			if self.cureProb > self.random.random():
				self.resistent += [self.disease]
				self.disease = 0
				self.sickTime = 0
				self.cureProb = 0.1
				print(self.resistent)
			else:
				self.cureProb *= 2

	def step(self):
		"""Move and spread disease if sick."""
		self.move()
		if self.disease >= 1:
			self.sickTime += 1
			self.mutate()
			self.spread_disease()
			self.cured()



model = DiseaseModel(10, 10, 10, 50, 50,[(0,0),(25,25),(50,50)])
for i in range(1000):
	model.step()


agent_counts = np.zeros((model.grid.width, model.grid.height))
for cell in model.grid.coord_iter():
	agent, x, y = cell
	if agent != None:
		agent_counts[x][y] = agent.disease
	else:
		agent_counts[x][y] = -1
plt.imshow(agent_counts, interpolation='nearest')
plt.colorbar()
plt.show()

for cell in model.grid.coord_iter():
	agent, x, y = cell
	if agent != None:
		agent_counts[x][y] = agent.goal[0]
	else:
		agent_counts[x][y] = -25
plt.imshow(agent_counts, interpolation='nearest')
plt.colorbar()
plt.show()
