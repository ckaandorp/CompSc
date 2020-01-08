from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.space import SingleGrid
import numpy as np
import matplotlib.pyplot as plt

class DiseaseModel(Model):
	"""A model with some number of agents."""
	def __init__(self, N, width, height):
		self.num_agents = N
		self.grid = SingleGrid(width, height, True)
		self.schedule = SimultaneousActivation(self)
		# Create agents
		for i in range(self.num_agents):
			a = MoneyAgent(i, self)
			self.schedule.add(a)
			# Add the agent to a random grid cell
			x = self.random.randrange(self.grid.width)
			y = self.random.randrange(self.grid.height)
			self.grid.place_agent(a, (x, y))

	def step(self):
		self.schedule.step()

class MoneyAgent(Agent):
	""" An agent with fixed initial wealth."""
	def __init__(self, unique_id, model):
		super().__init__(unique_id, model)
		self.disease = self.random.randrange(2)
		print(self.disease)

	def move(self):
		possible_steps = self.model.grid.get_neighborhood(
			self.pos,
			moore=False,
			include_center=True)
		new_position = self.random.choice(possible_steps)
		self.model.grid.move_agent(self, new_position)

	def spread_disease(self):
		cellmates = self.model.grid.get_cell_list_contents([self.model.grid.get_neighborhood(
			self.pos,
			moore=False,
			include_center=True)])
		if len(cellmates) > 1:
			other = self.random.choice(cellmates)
			other.disease = 1

	def step(self):
		self.move()
		if self.disease == 1:
			self.spread_disease()



model = DiseaseModel(50, 10, 10)
for i in range(20):
	model.step()


agent_counts = np.zeros((model.grid.width, model.grid.height))
for cell in model.grid.coord_iter():
	cell_content, x, y = cell
	agent_count = len(cell_content)
	agent_counts[x][y] = agent_count
plt.imshow(agent_counts, interpolation='nearest')
plt.colorbar()
plt.show()
