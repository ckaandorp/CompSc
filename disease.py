from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from mesa.space import SingleGrid
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from random import randint


def disease_spreader(cellmates, self, distanceFac):
	"""
	Calculates spread of disease to all cellmates.
	cellmates: list of all objects surrounding the agent
	self: current agent object
	distanceFac: factor to multiply the disease spreading rate with based on
				 distance between distance.
	"""
	if len(cellmates) > 0:
		# check all cellmates
		for i in range(len(cellmates)):
			other = cellmates[i]
			# ignore agents that are walls
			if not isinstance(other, wall) and not isinstance(self, wall):
				# check resistance of other agent
				if self.disease not in other.resistent:
					# disease will not be spread if a wall blocks the path
					if not wall_in_the_way(self, other):
					# ignore agents on other side of map
						if (abs(self.pos[0] - other.pos[0]) + abs(self.pos[1] - other.pos[1])) > 5:
							if self.model.diseaseRate * distanceFac > self.random.random():
								other.disease = self.disease

def wall_in_the_way(self, other):
	"""Returns True if there is a wall between agents, else false."""
	difference_x = self.pos[0] - other.pos[0]
	difference_y = self.pos[1] - other.pos[1]
	for i in range(abs(difference_x)):
		if difference_x < 0:
			i *= -1
		cell = self.model.grid.get_neighborhood((self.pos[0] + i,self.pos[1]), moore=False, include_center=True, radius=0)
		if cell != None and isinstance(cell,wall):
			return True
	for i in range(abs(difference_y)):
		if difference_y < 0:
			i *= -1
		cell = self.model.grid.get_neighborhood((self.pos[0]+difference_x, self.pos[1]+i), moore=False, include_center=True, radius=0)
		if cell != None and isinstance(cell,wall):
			return True
	return False

def disease_collector(model):
	"""
	Collects disease data from a model.
	Returns:
	the total percentage of agents that are sick,
	dictionary containting how many agents are suffering from each disease,
	number of different mutations and
	dictionary containing how many agents of each social group are sick.
	"""
	total_sick = 0
	disease_dict = {}
	social_dict = {'0':0, '1':0, '2':0}
	n_mutations = 0
	for agent in model.schedule.agents:
		# check if agent has a disease
		if agent.disease > 0:
			total_sick += 1
			social_dict[str(agent.sociability)] += 1
			# update number of mutations
			if agent.disease > n_mutations:
				n_mutations = agent.disease
			# add disease to disease dict if previously unknown
			if agent.disease in disease_dict:
				disease_dict[agent.disease] += 1
			else:
				disease_dict[agent.disease] = 1

	# calculate sick percentage per disease
	sum = 0
	for mutation in disease_dict:
		disease_dict[mutation] /= model.num_agents
		sum += disease_dict[mutation]

	return (total_sick/model.num_agents, disease_dict, n_mutations, social_dict)

def AStarSearch(start, end, graph):
	""" Code from: https://rosettacode.org/wiki/A*_search_algorithm#Python """
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
	edu_setting: Classrooms and set schedule if true, else random free movement.
	cureProb: Probability of agent getting better.
	cureProbFac: Factor of cureProb getting higher.
	mutateProb: Probability of a disease mutating.
	diseaseRate: Rate at which the disease spreads.
	"""
	def __init__(self, highS, middleS, lowS, width, height, edu_setting=True, cureProb=0.1, cureProbFac=2/1440, mutateProb=0.0005, diseaseRate=0.38):
		self.num_agents = highS + middleS + lowS
		self.lowS = lowS
		self.middleS = middleS
		self.highS = highS
		self.initialCureProb = cureProb
		self.cureProbFac = cureProbFac
		self.mutateProb = mutateProb
		self.diseaseRate = diseaseRate
		self.edu_setting = edu_setting
		self.maxDisease = 0 # amount of mutations
		self.counter = 0 # keeps track of timesteps

		# Check if agents fit within grid
		if self.num_agents > width * height:
			raise ValueError("Number of agents exceeds grid capacity.")

		# make grid with random activation.
		self.grid = SingleGrid(width, height, True)
		self.schedule = RandomActivation(self)

		if edu_setting:
			# Create walls
			numberRooms = 3
			self.add_walls(numberRooms, width, height)
			midWidthRoom = floor(width / numberRooms / 2)
			midHeightRoom = floor(height / numberRooms / 2)
			self.midWidthRoom = midWidthRoom
			self.midHeightRoom = midHeightRoom

			# Calculate the middlepoints of the 6 rooms
			roomLeftDown = (5 * midWidthRoom, midHeightRoom)
			roomLeftMid = (3 * midWidthRoom, midHeightRoom)
			roomLeftUp = (midWidthRoom, midHeightRoom)
			roomRightDown = (5 * midWidthRoom, 5 * midHeightRoom, )
			roomRightMid = (3 * midWidthRoom, 5 * midHeightRoom)
			roomRightUp = (midWidthRoom, 5 * midHeightRoom)

			# Set goals 
			self.roster = [[roomLeftDown, roomLeftUp, roomRightMid], [roomRightMid, roomLeftDown, roomRightDown], 
							[roomRightUp, roomRightDown, roomLeftUp]]

		# Create agents
		self.addAgents(lowS, 0, 0)
		self.addAgents(middleS, lowS, 1)
		self.addAgents(highS, lowS + highS, 2)

		self.datacollector = DataCollector(
			model_reporters={"diseasepercentage": disease_collector}, 
			agent_reporters={"disease": "disease"})

	def heuristic(self, start, goal):
		""" 
		Returns manhattan distance.
		start: current location (x,y)
		goal: goal location (x,y)
		"""
		dx = abs(start[0] - goal[0])
		dy = abs(start[1] - goal[1])
		return dx + dy

	def get_vertex_neighbours(self, pos):
		"""
		Returns all neighbors.
		pos: current position
		"""
		n = self.grid.get_neighborhood(pos, moore=False)
		neighbors = []
		for item in n:
			if not abs(item[0]-pos[0]) > 1 and not abs(item[1]-pos[1]) > 1:
				neighbors += [item]
		return neighbors

	def move_cost(self, a, b):
		if self.grid.is_cell_empty(b):
			return 1 # Normal movement cost
		else:
			return 100

	def add_walls(self, n, widthGrid, heightGrid):
		""" 
		Add walls in grid.
		n: number of rooms vertically
		widthGrid: width of the grid
		heightGrid: height of the grid
		"""
		widthRooms = floor(widthGrid/n)
		heightRooms = floor(heightGrid/n)
		widthHall = widthGrid - 2 * widthRooms
		heightHall = heightGrid - 2 * heightRooms
		# Add horizontal walls
		for i in range(n - 1):
			for y in range(heightRooms):
				brick = wall(self.num_agents, self)
				self.grid.place_agent(brick, ((i + 1) * widthRooms, y))
				self.grid.place_agent(brick, ((i + 1) * widthRooms, y + heightRooms + heightHall))
		doorWidth = 2
		# Add vertical walls
		for x in range(widthGrid):
			if (x % widthRooms) < (widthRooms - doorWidth):
				brick = wall(self.num_agents, self)
				self.grid.place_agent(brick, (x, heightRooms))
				self.grid.place_agent(brick, (x, heightRooms + heightHall - 1))

	def addAgents(self, n, startID, sociability):
		""" 
		Add agents with a sociability. 
		n: number of agents
		startID: ID of the first added agent
		sociability: sociability of the agents
		"""
		for i in range(n):
			a = DiseaseAgent(i + startID, sociability, self)
			self.schedule.add(a)
			# Add the agent to a random grid cell
			location = self.grid.find_empty()
			self.grid.place_agent(a, location)

	def step(self):
		""" Continue one step in simulation. """
		self.counter += 1
		self.datacollector.collect(self)
		self.schedule.step()

class DiseaseAgent(Agent):
	""" An agent with fixed initial disease."""
	def __init__(self, unique_id, sociability, model):
		super().__init__(unique_id, model)
		# Randomly set agent as healthy or sick
		self.disease = self.random.randrange(2)
		self.sociability = sociability
		self.resistent = []
		self.cureProb = self.model.initialCureProb
		self.sickTime = 0
		self.talking = 0.1
		self.path = []
		if self.model.edu_setting == True:
			self.roster = self.model.roster[self.random.randrange(len(self.model.roster))]
			self.goal = (self.roster[0][0] + randint(-self.model.midWidthRoom - 1, self.model.midWidthRoom - 1), self.roster[0][1] + randint(-self.model.midHeightRoom - 1, self.model.midHeightRoom - 1))

	def random_move(self):
		""" Moves agent one step on the grid."""
		possible_steps = self.model.grid.get_neighborhood(
			self.pos,
			moore=False,
			include_center=True)
		possible_steps_real = []
		for cell in possible_steps:
			if not abs(cell[0]-self.pos[0]) > 1 and not abs(cell[1]-self.pos[1]) > 1:
				possible_steps_real += [cell]
		choice = self.random.choice(possible_steps_real)
		if self.model.grid.is_cell_empty(choice):
			self.model.grid.move_agent(self, choice)

	def move(self):
		""" 
		Moves agent one step on the grid.
		"""
		if self.model.counter > 200:
			self.goal = (self.roster[1][0] + randint(-self.model.midWidthRoom - 1, self.model.midWidthRoom - 1), self.roster[1][1] + randint(-self.model.midHeightRoom - 1, self.model.midHeightRoom - 1))
			self.path = []
		elif self.model.counter > 400:
			self.goal = (self.roster[2][0] + randint(-self.model.midWidthRoom - 1, self.model.midWidthRoom - 1), self.roster[2][1] + randint(-self.model.midHeightRoom - 1, self.model.midHeightRoom - 1))
			self.path = []

		if not isinstance(self, wall):
			cellmates = self.model.grid.get_neighbors(self.pos, moore=True)
			newCellmates = []
			for cellmate in cellmates:
				if not abs(cellmate.pos[0]-self.pos[0]) > 1 and not abs(cellmate.pos[1]-self.pos[1]) > 1 and not isinstance(cellmate, wall):
					newCellmates += [cellmate]

			# behavior based on sociability.
			# move away from agent if low sociability
			if self.sociability == 0:
				if len(newCellmates) > 0:
					other = self.random.choice(newCellmates)
					# find escape route
					escape = ((self.pos[0] - other.pos[0]), (self.pos[1] - other.pos[1]))
					choice = (escape[0] + self.pos[0], escape[1] + self.pos[1])
					if self.model.grid.width > choice[0] > 0 and self.model.grid.height > choice[1] > 0:
						if self.model.grid.is_cell_empty(choice):
							self.model.grid.move_agent(self, choice)
							return
			# stop if talked to if middle sociability
			if self.sociability == 1 and self.random.random() > self.talking:
				for neighbor in newCellmates:
					if neighbor.sociability == 2:
						self.talking *= 2
						return
			else:
				self.talking = 0.1

			# stop to talk if there is a neighbor if high sociability
			if self.sociability == 2  and self.random.random() > self.talking:
				if len(cellmates) > 0:
					self.talking *= 2
					return
			else:
				self.talking = 0.1

			if self.pos != self.goal:
				if self.path == []:
					self.path = AStarSearch(self.pos, self.goal, model)
				if self.path != []:
					if self.path != [-1] and model.grid.is_cell_empty(self.path[0]):
						self.model.grid.move_agent(self,self.path[0])
						self.path.pop(0)
					else:
						self.path = AStarSearch(self.pos, self.goal, model)
						if self.path != [-1] and model.grid.is_cell_empty(self.path[0]):
							self.model.grid.move_agent(self, self.path[0])
							self.path.pop(0)

	def spread_disease(self):
		"""Spreads disease to neighbors."""
		cellmates = set(self.model.grid.get_neighbors(self.pos,moore=True))
		cellmates_2 = set(self.model.grid.get_neighbors(self.pos,moore=True,radius=2))
		cellmates_3 = set(self.model.grid.get_neighbors(self.pos,moore=True,radius=3))
		cellmates_4 = set(self.model.grid.get_neighbors(self.pos,moore=True,radius=4))
		cellmates = list(cellmates)
		cellmates_2 = list(cellmates_2.difference(cellmates))
		cellmates_3 = list(cellmates_3.difference(cellmates_2))
		cellmates_4 = list(cellmates_4.difference(cellmates_3))
		# Check if there are neighbors to spread disease to
		disease_spreader(cellmates,self,1)
		disease_spreader(cellmates_2,self,0.75)
		disease_spreader(cellmates_3,self,0.5)
		disease_spreader(cellmates_4,self,0.125)

	def mutate(self):
		"""Mutates disease in an agent."""
		if self.disease > 0:
			if self.model.mutateProb > self.random.random():
				self.model.maxDisease += 1
				self.disease = self.model.maxDisease

	def cured(self):
		"""Cure agents based on cure probability sick time."""
		if self.sickTime > 10080: # people are generally sick for at least 1 week (60 * 24 * 7 = 10080)
			# Agent is cured
			if self.cureProb > self.random.random():
				self.resistent += [self.disease]
				self.disease = 0
				self.sickTime = 0
				self.cureProb = self.model.initialCureProb
			else:
				self.cureProb *= self.model.cureProbFac

	def step(self):
		"""Move and spread disease if sick."""
		if self.model.edu_setting == False:
			self.random_move()
		else:
			self.move()
		if self.disease >= 1:
			self.sickTime += 1
			self.mutate()
			self.spread_disease()
			self.cured()


class wall(Agent):
	"""A wall seperating the spaces."""
	def __init__(self, unique_id, model):
		self.disease = -1
		super().__init__(unique_id, model)


def disease_graph(model):
	""""Plots progress of disease given a model."""
	# get dataframe
	df = model.datacollector.get_model_vars_dataframe()
	diseased = []
	mutation = []
	low_sociability = []
	middle_sociability = []
	high_sociability = []
	n_mutations = 0

	for index, row in df.iterrows():
		diseased += [row[0][0]]
		mutation += [row[0][1]]
		sociability = row[0][3]
		low_sociability += [sociability['0']]
		middle_sociability += [sociability['1']]
		high_sociability += [sociability['2']]
		if row[0][2] > n_mutations:
			n_mutations = row[0][2]

	plt.plot(diseased, color="red", label='total')

	# collect all diseases
	disease_plotter = []

	for _ in range(n_mutations):
		disease_plotter += [[]]
	for j in range(len(mutation)):
		for i in range(n_mutations):
			if i+1 in mutation[j]:
				disease_plotter[i] += [mutation[j][i+1]]
			else:
				disease_plotter[i] += [0]
	# plot all diseases
	for mutation in disease_plotter:
		plt.plot(mutation) #linestyle='dashed')

	plt.xlabel('Timesteps')
	plt.ylabel('Infected (%)')
	plt.legend()
	plt.show()

	# plot agent sociability
	plt.plot([x / model.lowS for x in low_sociability], label='low ' + str(model.lowS))
	plt.plot([x / model.middleS for x in middle_sociability], label='middle ' + str(model.middleS))
	plt.plot([x / model.highS for x in high_sociability], label='high ' + str(model.highS))
	plt.ylabel("Infected (%)")
	plt.xlabel("Timesteps")
	plt.legend()
	plt.show()




# model = DiseaseModel(10, 10, 10, 50, 50, edu_setting=False, mutateProb=0.005)
#
# for i in range(300):
# 	print(i)
# 	model.step()

#
# agent_counts = np.zeros((model.grid.width, model.grid.height))
# for cell in model.grid.coord_iter():
# 	agent, x, y = cell
# 	if agent != None and not isinstance(agent, wall):
# 		agent_counts[x][y] = agent.disease
# 	elif agent != None and isinstance(agent, wall):
# 		agent_counts[x][y] = 5
# 	else:
# 		agent_counts[x][y] = -1
# plt.imshow(agent_counts, interpolation='nearest')
# plt.colorbar()
# plt.show()
#
#
#
# disease_graph(model)
