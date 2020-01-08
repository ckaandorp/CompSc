from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
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
	def __init__(self, highS, middleS, lowS, width, height):
		self.num_agents = highS + middleS + lowS
		if self.num_agents > width * height:
			raise ValueError("Number of agents exceeds grid capacity.")

		self.grid = SingleGrid(width, height, True)
		self.schedule = RandomActivation(self)
		# Create agents
		for i in range(lowS):
			a = DiseaseAgent(i, 0, self)
			self.schedule.add(a)
			# Add the agent to a random grid cell
			location = self.grid.find_empty()
			self.grid.place_agent(a, location)
		for i in range(middleS):
			a = DiseaseAgent(i, 1, self)
			self.schedule.add(a)
			# Add the agent to a random grid cell
			location = self.grid.find_empty()
			self.grid.place_agent(a, location)
		for i in range(highS):
			a = DiseaseAgent(i, 2, self)
			self.schedule.add(a)
			# Add the agent to a random grid cell
			location = self.grid.find_empty()
			self.grid.place_agent(a, location)
		self.datacollector = DataCollector(
			model_reporters={"Gini": "grid"},
			agent_reporters={"Disease": "disease"})
	# Continue one step in simulation
	def step(self):
		self.schedule.step()
		self.datacollector.collect(self)
class DiseaseAgent(Agent):
	""" An agent with fixed initial disease."""
	def __init__(self, unique_id, sociability, model):
		super().__init__(unique_id, model)
		# Randomly set agent as healthy or sick
		self.disease = self.random.randrange(2)
		self.diseaserate = 0.7
		self.sociability = sociability
		self.resistent = 0

	def move(self):
		""" Moves agent one step on the grid."""
		possible_steps = self.model.grid.get_neighborhood(
			self.pos,
			moore=False,
			include_center=True)
		choice = self.random.choice(possible_steps)
		if model.grid.is_cell_empty(choice):
			self.model.grid.move_agent(self, choice)


	def spread_disease(self):
		"""Spreads disease to neighbors."""
		cellmates = self.model.grid.get_neighbors(self.pos,moore=True)
		if len(cellmates) > 1:
			for i in range(len(cellmates)):
				other = cellmates[i]
				if other.resistent == 0:
					if self.diseaserate > self.random.random():
						other.disease = 1
	def cured(self):
		if 1 > self.random.random():
			self.disease = 0
			self.resistent = 1
			print(self.resistent)
	def step(self):
		"""Move and spread disease if sick."""
		self.move()
		print(self.disease)
		if self.disease == 1:
			self.spread_disease()
			self.cured()



model = DiseaseModel(10, 10, 10, 10, 10)
for i in range(200):
	model.step()

test = model.datacollector.get_model_vars_dataframe()
grid = test['Gini'][199]
print(grid)

agent_counts = np.zeros((model.grid.width, model.grid.height))
for cell in grid.coord_iter():
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
		agent_counts[x][y] = agent.resistent
	else:
		agent_counts[x][y] = -1
plt.imshow(agent_counts, interpolation='nearest')
plt.colorbar()
plt.show()
