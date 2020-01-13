from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from mesa.space import SingleGrid
import numpy as np
import matplotlib.pyplot as plt


def diseasecolletor(model):
	disease_counter = 0
	disease_dict = {}
	max_disease = 0
	for agent in model.schedule.agents:
		if agent.disease > 0:
			disease_counter += 1
			if agent.disease > max_disease:
				max_disease = agent.disease
			if agent.disease in disease_dict:
				disease_dict[agent.disease] += 1
			else:
				disease_dict[agent.disease] = 1
	for mutation in disease_dict:
		disease_dict[mutation] /= model.num_agents
	return (disease_counter/len(model.schedule.agents),disease_dict,max_disease)

def AStarSearch(start, end, graph):

	G = {} #Actual movement cost to each position from the start position
	F = {} #Estimated movement cost of start to end going via this position

	#Initialize starting values
	G[start] = 0
	F[start] = graph.heuristic(start, end)

	closedVertices = set()
	openVertices = set([start])
	cameFrom = {}

	while len(openVertices) > 0:
		#Get the vertex in the open list with the lowest F score
		current = None
		currentFscore = None
		for pos in openVertices:
			if current is None or F[pos] < currentFscore:
				currentFscore = F[pos]
				current = pos

		#Check if we have reached the goal
		if current == end:
			#Retrace our route backward
			path = [current]
			while current in cameFrom:
				current = cameFrom[current]
				path.append(current)
			path.reverse()
			return path[1:] #Done!

		#Mark the current vertex as closed
		openVertices.remove(current)
		closedVertices.add(current)

		#Update scores for vertices near the current position
		for neighbour in graph.get_vertex_neighbours(current):
			if neighbour in closedVertices:
				continue #We have already processed this node exhaustively
			candidateG = G[current] + graph.move_cost(current, neighbour)

			if neighbour not in openVertices:
				openVertices.add(neighbour) #Discovered a new vertex
			elif candidateG >= G[neighbour]:
				continue #This G score is worse than previously found

			#Adopt this G score
			cameFrom[neighbour] = current
			G[neighbour] = candidateG
			H = graph.heuristic(neighbour, end)
			F[neighbour] = G[neighbour] + H
	return [-1]

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
		#check if agent count is heigh then squares.
		if self.num_agents > width * height:
			raise ValueError("Number of agents exceeds grid capacity.")

		#make grid with random activation.
		self.grid = SingleGrid(width, height, True)
		self.schedule = RandomActivation(self)

		# Create agents
		self.addAgents(lowS, 0, 0)
		self.addAgents(middleS, lowS, 1)
		self.addAgents(highS, lowS + highS, 2)

		self.datacollector = DataCollector(
		model_reporters={"diseasepercentage": diseasecolletor},  # `compute_gini` defined above
		agent_reporters={"disease": "disease"})

	def heuristic(self, start, goal):
		#manhatan distance
		dx = abs(start[0] - goal[0])
		dy = abs(start[1] - goal[1])
		return dx+dy

	def get_vertex_neighbours(self, pos):
		n = self.grid.get_neighborhood(pos, moore=False)
		for item in n:
			if abs(item[0]-pos[0]) > 1 or abs(item[1]-pos[1]) > 1:
				n.remove(item)
		#Moves allow link a chess king
		return n

	def move_cost(self, a, b):
		# for barrier in self.barriers:
		# 	if b in barrier:
		# 		return 100 #Extremely high cost to enter barrier squares
		if model.grid.is_cell_empty(b):
			return 1 #Normal movement cost
		else:
			return 100


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
		self.cureProb = 0.1
		self.sickTime = 0
		self.path = []
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

		if self.path == []:
			self.path = AStarSearch(self.pos, self.goal, model)
		if self.path != []:
			if self.path != [-1] and model.grid.is_cell_empty(self.path[0]):
				self.model.grid.move_agent(self,self.path[0])
				self.path.pop(0)
			else:
				self.path = AStarSearch(self.pos, self.goal, model)
				if self.path != [-1] and model.grid.is_cell_empty(self.path[0]):
					self.model.grid.move_agent(self,self.path[0])
					self.path.pop(0)
		# # goal based movement
		# x_distance = self.goal[0] - self.pos[0]
		# y_distance = self.goal[1] - self.pos[1]
		# # takes a step in the direction that is farthest off the current position.
		# # if more horizontal distance needs to be traveled.
		# if abs(x_distance) >= abs(y_distance):
		# 	if x_distance > 0:
		# 		choice = (self.pos[0]+1,self.pos[1])
		# 		if 0 < choice[0] < model.grid.width and  0 < choice[1] < model.grid.height and model.grid.is_cell_empty(choice):
		# 			self.model.grid.move_agent(self,choice)
		# 	else:
		# 		choice = (self.pos[0]-1,self.pos[1])
		# 		if 0 < choice[0] < model.grid.width and  0 < choice[1] < model.grid.height and model.grid.is_cell_empty(choice):
		# 			self.model.grid.move_agent(self,choice)
		# # if more vertical distance needs be traveled.
		# else:
		# 	if y_distance > 0:
		# 		choice = (self.pos[0],self.pos[1]+1)
		# 		if 0 < choice[0] < model.grid.width and  0 < choice[1] < model.grid.height and model.grid.is_cell_empty(choice):
		# 			self.model.grid.move_agent(self,choice)
		# 	else:
		# 		choice = (self.pos[0],self.pos[1]-1)
		# 		if 0 < choice[0] < model.grid.width and  0 < choice[1] < model.grid.height and model.grid.is_cell_empty(choice):
		# 			self.model.grid.move_agent(self,choice)

	def spread_disease(self):
		"""Spreads disease to neighbors."""
		cellmates = self.model.grid.get_neighbors(self.pos,moore=True)
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

	def mutate(self):
		"""mutates disease in an agent."""
		if self.disease > 0:
			if 0.1 > self.random.random():
				self.disease += 1

	def cured(self):
		"""Cure agents based on cure probability sick time."""
		if self.sickTime > 10080: # people are generally sick for at least 1 week (60 * 24 * 7)
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


model = DiseaseModel(10, 10, 10, 25, 25,[(0,0),(12,12),(24,24)])

for i in range(50):
	print(i)
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
		agent_counts[x][y] = -5
plt.imshow(agent_counts, interpolation='nearest')
plt.colorbar()
plt.show()

df = model.datacollector.get_model_vars_dataframe()
diseased = []
mutation = []
max_disease = 0
print()
for index, row in df.iterrows():
	diseased += [row[0][0]]
	mutation += [row[0][1]]
	if row[0][2] > max_disease:
		max_disease = row[0][2]


plt.plot(diseased,color="red")


disease_plotter = []
for i in range(max_disease):
	disease_plotter += [[]]
for j in range(len(mutation)):
	for i in range(max_disease):
		if i in mutation[j]:
			disease_plotter[i] += [mutation[j][i]]
		else:
			disease_plotter[i] += [0]

for mutation in disease_plotter:
	plt.plot(mutation)

plt.show()
