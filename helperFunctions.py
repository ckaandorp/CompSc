from wall import wall

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
	resistent_dict = {'0':0, '1':0, '2':0}
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
		resistent_dict[str(agent.sociability)] += len(agent.resistent)
	# calculate sick percentage per disease
	sum = 0
	for mutation in disease_dict:
		disease_dict[mutation] /= model.num_agents
		sum += disease_dict[mutation]


	return (total_sick/model.num_agents, disease_dict, n_mutations, social_dict,resistent_dict)

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
		for neighbor in graph.get_vertex_neighbors(current):
			if neighbor in closedVertices:
				continue #We have already processed this node exhaustively
			candidateG = G[current] + graph.move_cost(neighbor)

			if neighbor not in openVertices:
				openVertices.add(neighbor) #Discovered a new vertex
			elif candidateG >= G[neighbor]:
				continue #This G score is worse than previously found

			#Adopt this G score
			cameFrom[neighbor] = current
			G[neighbor] = candidateG
			H = graph.heuristic(neighbor, end)
			F[neighbor] = G[neighbor] + H
	return [-1]
