from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from mesa.space import SingleGrid
import numpy as np
from math import floor
from DiseaseAgent import DiseaseAgent
from Wall import wall
from helperFunctions import disease_collector

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
	def __init__(self, highS, middleS, lowS, width, height, edu_setting=True, cureProb=0.1, cureProbFac=2/1440, mutateProb=0.0050, diseaseRate=0.38):
		super().__init__()
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


model = DiseaseModel(10, 10, 10, 50, 50, edu_setting=False, mutateProb=0.005)

for i in range(300):
	print(i)
	model.step()

disease_graph(model)

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
