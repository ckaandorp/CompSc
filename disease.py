from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import SingleGrid
import numpy as np
import matplotlib.pyplot as plt

class DiseaseModel(Model):
	"""A model with some number of agents."""
	def __init__(self, N, width, height):
		self.num_agents = N
		self.grid = SingleGrid(width, height, True)
		self.schedule = RandomActivation(self)
		# Create agents
		for i in range(self.num_agents):
			a = DiseaseAgent(i, self)
			self.schedule.add(a)
			# Add the agent to a random grid cell
			location = self.grid.find_empty()
			self.grid.place_agent(a, location)

	# Continue one step in simulation
	def step(self):
		self.schedule.step()

class DiseaseAgent(Agent):
	""" An agent with fixed initial wealth."""
	def __init__(self, unique_id, model):
		super().__init__(unique_id, model)
		# Randomly set agent as healthy or sick
		self.disease = self.random.randrange(2)
		self.diseaserate = 0.7

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
				if self.diseaserate > self.random.random():
					other.disease = 1
	def cured(self):
		if 0.4 > self.random.random():
			self.disease = 0
	def step(self):
		"""Move and spread disease if sick."""
		self.move()
		if self.disease == 1:
			self.spread_disease()
			self.cured()



model = DiseaseModel(50, 10, 10)
for i in range(20):
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
