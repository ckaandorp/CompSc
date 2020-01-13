from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from mesa.space import SingleGrid
import numpy as np
import matplotlib.pyplot as plt


def disease_collector(model):
	""" 
	Collects disease data from a model.
	Returns: 
	the total percentage of agents that are sick,
	dictionary containting how many agents are suffering from each disease and
	number of different mutations.
	"""
	total_sick = 0
	disease_dict = {}
	n_mutations = 0
	for agent in model.schedule.agents:
		# check if agent has a disease
		if agent.disease > 0:
			total_sick += 1
			# update number of mutations
			if agent.disease > n_mutations:
				n_mutations = agent.disease
			# add disease to disease dict if previously unknown
			if agent.disease in disease_dict:
				disease_dict[agent.disease] += 1
			else:
				disease_dict[agent.disease] = 1
	# calculate sick percentage per disease
	for mutation in disease_dict:
		disease_dict[mutation] /= model.num_agents
		
	return (total_sick/model.num_agents, disease_dict, n_mutations)

class DiseaseModel(Model):
	"""
	A model with some number of agents.
	highS: Number of agents with high sociability.
	middleS: Number of agents with middle sociability.
	lowS: Number of agents with low sociability.
	width: Width of the grid.
	height: Height of the grid.
	"""
	def __init__(self, highS, middleS, lowS, width, height, rooms, cureProb=0.1, cureProbFac=2, mutateProb=0.1):
		self.num_agents = highS + middleS + lowS
		self.rooms = rooms
		self.initialCureProb = cureProb
		self.cureProbFac = cureProbFac
		self.mutateProb = mutateProb
		# Check if agent fit within grid
		if self.num_agents > width * height:
			raise ValueError("Number of agents exceeds grid capacity.")

		# make grid with random activation.
		self.grid = SingleGrid(width, height, True)
		self.schedule = RandomActivation(self)

		# Create agents
		self.addAgents(lowS, 0, 0)
		self.addAgents(middleS, lowS, 1)
		self.addAgents(highS, lowS + highS, 2)

		self.datacollector = DataCollector(
		model_reporters={"diseasepercentage": disease_collector},  # `compute_gini` defined above
		agent_reporters={"disease": "disease"})

	def addAgents(self, n, startID, sociability):
		# add n amount of agents with a sociability
		for i in range(n):
			a = DiseaseAgent(i + startID, sociability, self)
			self.schedule.add(a)
			# Add the agent to a random grid cell
			location = self.grid.find_empty()
			self.grid.place_agent(a, location)


	# Continue one step in simulation
	def step(self):
		self.datacollector.collect(self)
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
		self.initialCureProb = self.model.initialCureProb
		self.cureProb = self.initialCureProb
		self.cureProbFac = self.model.cureProbFac
		self.mutateProb = self.model.mutateProb
		self.sickTime = 0
		self.goal = self.model.rooms[self.random.randrange(len(self.model.rooms))]

	def move(self):
		""" Moves agent one step on the grid."""
		cellmates = self.model.grid.get_neighbors(self.pos, moore=True)

		# behavior based on sociability.
		# move away from agent if low sociability
		if self.sociability == 0:
			if len(cellmates) > 0:
				other = self.random.choice(cellmates)
				# find escape route
				escape = ((self.pos[0] - other.pos[0]), (self.pos[1] - other.pos[1]))
				choice = (escape[0] + self.pos[0], escape[1] + self.pos[1])
				if self.model.grid.width > choice[0] > 0 and self.model.grid.height > choice[1] > 0:
					if model.grid.is_cell_empty(choice):
						self.model.grid.move_agent(self, choice)
						return
		# stop if talked to if middle sociability
		if self.sociability == 1:
			for neighbor in cellmates:
				if neighbor.sociability == 2:
					return
		# stop to talk if there is a neighbor if high sociability
		if self.sociability == 2:
			if len(cellmates) > 0:
				return

		# goal based movement
		x_distance = self.goal[0] - self.pos[0]
		y_distance = self.goal[1] - self.pos[1]
		# takes a step in the direction that is farthest off the current position.
		# if more horizontal distance needs to be traveled.
		if abs(x_distance) >= abs(y_distance):
			if x_distance > 0:
				choice = (self.pos[0]+1,self.pos[1])
				if 0 < choice[0] < model.grid.width and  0 < choice[1] < model.grid.height and model.grid.is_cell_empty(choice):
					self.model.grid.move_agent(self,choice)
			else:
				choice = (self.pos[0]-1,self.pos[1])
				if 0 < choice[0] < model.grid.width and  0 < choice[1] < model.grid.height and model.grid.is_cell_empty(choice):
					self.model.grid.move_agent(self,choice)
		# if more vertical distance needs be traveled.
		else:
			if y_distance > 0:
				choice = (self.pos[0],self.pos[1]+1)
				if 0 < choice[0] < model.grid.width and  0 < choice[1] < model.grid.height and model.grid.is_cell_empty(choice):
					self.model.grid.move_agent(self,choice)
			else:
				choice = (self.pos[0],self.pos[1]-1)
				if 0 < choice[0] < model.grid.width and  0 < choice[1] < model.grid.height and model.grid.is_cell_empty(choice):
					self.model.grid.move_agent(self,choice)

	def spread_disease(self):
		"""Spreads disease to neighbors."""
		cellmates = self.model.grid.get_neighbors(self.pos,moore=True)
		# Check if there are neighbors to spread disease to
		if len(cellmates) > 0:
			for i in range(len(cellmates)):
				other = cellmates[i]
				if self.disease not in other.resistent:
					# if direct neighbor then normal infection probability, else lowered.
					if (abs(self.pos[0] - other.pos[0]) + abs(self.pos[1] - other.pos[1])) == 1:
						if self.diseaserate > self.random.random():
							other.disease = self.disease
					else:
						if (self.diseaserate * 0.75) > self.random.random():
							other.disease = self.disease

	def mutate(self,):
		"""Mutates disease in an agent."""
		if self.disease > 0:
			if self.mutateProb > self.random.random():
				self.disease += 1

	def cured(self):
		"""Cure agents based on cure probability sick time."""
		if self.sickTime > 10080: # people are generally sick for at least 1 week (60 * 24 * 7 = 10080)
			# Agent is cured
			if self.cureProb > self.random.random():
				self.resistent += [self.disease]
				self.disease = 0
				self.sickTime = 0
				self.cureProb = self.initialCureProb 
				print(self.resistent)
			else:
				self.cureProb *= self.cureProbFac

	def step(self):
		"""Move and spread disease if sick."""
		self.move()
		if self.disease >= 1:
			self.sickTime += 1
			self.mutate()
			self.spread_disease()
			self.cured()



model = DiseaseModel(10, 10, 10, 50, 50,[(0,0),(12,25),(50,50)])
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

for cell in model.grid.coord_iter():
	agent, x, y = cell
	if agent != None:
		agent_counts[x][y] = agent.goal[0]
	else:
		agent_counts[x][y] = -25
plt.imshow(agent_counts, interpolation='nearest')
plt.colorbar()
plt.show()

# get dataframe
df = model.datacollector.get_model_vars_dataframe()
diseased = []
mutation = []
n_mutations = 0
print()
for index, row in df.iterrows():
	diseased += [row[0][0]]
	mutation += [row[0][1]]
	if row[0][2] > n_mutations:
		n_mutations = row[0][2]


plt.plot(diseased,color="red")


disease_plotter = []
for i in range(n_mutations):
	disease_plotter += [[]]
for j in range(len(mutation)):
	for i in range(n_mutations):
		if i in mutation[j]:
			disease_plotter[i] += [mutation[j][i]]
		else:
			disease_plotter[i] += [0]

for mutation in disease_plotter:
	plt.plot(mutation)

plt.show()
