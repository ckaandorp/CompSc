from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from mesa.space import SingleGrid
import numpy as np
from math import floor
from diseaseAgent import DiseaseAgent
from wall import wall
from helperFunctions import disease_collector


class DiseaseModel(Model):
    """
    A model with some number of agents.
    highS: Number of agents with high sociability.
    middleS: Number of agents with middle sociability.
    lowS: Number of agents with low sociability.
    width: Width of the grid.
    height: Height of the grid.
    edu_setting: If true, agents will follow a schedule and sit in classrooms,
    else they will move freely through an open grid.
    cureProb: Probability of agent getting better.
    cureProbFac: Factor of cureProb getting higher.
    mutateProb: Probability of a disease mutating.
    diseaseRate: Rate at which the disease spreads.
    """
    def __init__(self, highS, middleS, lowS, width, height, edu_setting=True,
                 cureProb=0.1, cureProbFac=2/1440, mutateProb=0.0050,
                 diseaseRate=0.38):
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
        self.maxDisease = 0  # amount of mutations
        self.counter = 540  # keeps track of timesteps
        self.removed = []
        self.exit = (width - 1, floor(height / 2))
        # Check if agents fit within grid
        if self.num_agents > width * height:
            raise ValueError("Number of agents exceeds grid capacity.")

        # Create grid with random activation
        self.grid = SingleGrid(width, height, True)
        self.schedule = RandomActivation(self)

        if edu_setting:
            # Create walls
            numberRooms = 3
            self.add_walls(numberRooms, width, height)

            self.midWidthRoom = floor(width / numberRooms / 2)
            self.midHeightRoom = floor(height / numberRooms / 2)
            self.widthRoom = floor(width / numberRooms)
            self.heightRoom = floor(height / numberRooms)
            numberRows = floor((self.heightRoom) / 2)
            widthRows = self.widthRoom - 4
            location = [[] for _ in range(numberRooms * 2)]
            for i in range(numberRooms):
                for j in range(0, numberRows, 2):
                    startWidth = 2 + (i % 3) * self.widthRoom
                    for currentWidth in range(widthRows):
                        location[i] += [(startWidth + currentWidth, j)]
            for i in range(3, numberRooms * 2):
                for j in range(0, numberRows, 2):
                    startWidth = 2 + (i % 3) * self.widthRoom
                    for currentWidth in range(widthRows):
                        location[i] += [(startWidth + currentWidth,
                                         height - 1 - j)]

            # Set 3 goals per roster
            self.roster = [[location[0], location[3], location[1]],
                           [location[5], location[2], location[0]],
                           [location[4], location[1], location[5]]]

        # Create agents
        self.addAgents(lowS, 0, 0)
        self.addAgents(middleS, lowS, 1)
        self.addAgents(highS, lowS + highS, 2)

        # set up data collecter
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

    def get_vertex_neighbors(self, pos):
        """
        Returns all neighbors.
        pos: current position
        """
        n = self.grid.get_neighborhood(pos, moore=False)
        neighbors = []
        for item in n:
            if not abs(item[0] - pos[0]) > 1 and not abs(item[1] - pos[1]) > 1:
                neighbors += [item]
        return neighbors

    def move_cost(self, location):
        """
        Return the cost of a location.
        """
        if self.grid.is_cell_empty(location):
            return 1  # Normal movement cost
        else:
            return 100  # Very difficult to go through walls

    def add_walls(self, n, widthGrid, heightGrid):
        """
        Add walls in grid.
        n: number of rooms horizontally
        widthGrid: width of the grid
        heightGrid: height of the grid
        """
        widthRooms = floor(widthGrid / n)
        heightRooms = floor(heightGrid / n)
        heightHall = heightGrid - 2 * heightRooms
        # Add horizontal walls
        for i in range(n - 1):
            for y in range(heightRooms):
                brick = wall(self.num_agents, self)
                self.grid.place_agent(brick, ((i + 1) * widthRooms, y))
                self.grid.place_agent(brick, ((i + 1) * widthRooms, y +
                                      heightRooms + heightHall))
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
        disease_list = np.random.randint(0, 2, n)
        for i in range(n):
            # Set schedule for every agent if educational setting
            if self.edu_setting:
                a_roster = []
                rosterNumber = self.random.randrange(len(self.roster))
                rooms = self.roster[rosterNumber]
                for roomNumber in range(len(rooms)):
                    loc = self.random.choice(rooms[roomNumber])
                    a_roster += [loc]
                    (self.roster[rosterNumber][roomNumber]).remove(loc)
            else:
                a_roster = []

            a = DiseaseAgent(i + startID, sociability, self, disease_list[i],
                             a_roster)
            self.schedule.add(a)
            # Set agent outside grid, ready to enter, if edu setting
            # else randomly place on empty spot on grid
            if self.edu_setting:
                self.removed += [a]
                a.pos = None
            else:
                self.grid.place_agent(a, self.grid.find_empty())

    def step(self):
        """
        Continue one step in simulation.
        """
        self.counter += 1
        self.datacollector.collect(self)
        self.schedule.step()
