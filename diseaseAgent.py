from mesa import Agent
from wall import wall
from helperFunctions import *


class DiseaseAgent(Agent):
    """
    An agent with fixed initial disease.
    unique_id: id of the agent
    sociability: social level of the agent
    model: model that the agent is in
    disease: > 0 if agent has a disease, 0 if agent is healthy
    roster: list of tuples containing goals for the agent
    """
    def __init__(self, unique_id, sociability, model, disease, roster):
        super().__init__(unique_id, model)
        self.id = unique_id
        self.disease = disease
        self.sociability = sociability
        self.resistant = []
        self.cureProb = self.model.initialCureProb
        self.sickTime = 0
        self.talking = 0.1  # probability that an agent stop talking
        self.path = []
        self.goal = 0
        self.talkedto = False
        self.roster = roster

    def random_move(self):
        """
        Moves agent one step on the grid freely (non-educational setting).
        """
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=False,
            include_center=True)
        possible_steps_real = []
        for cell in possible_steps:
            if not abs(cell[0]-self.pos[0]) > 1 and \
             not abs(cell[1]-self.pos[1]) > 1:
                possible_steps_real += [cell]
        choice = self.random.choice(possible_steps_real)
        if self.model.grid.is_cell_empty(choice):
            self.model.grid.move_agent(self, choice)

    def move(self):
        """
        Moves agent one step on the grid according to schedule
        (educational setting).
        """
        # End of schoolday, move towards exit
        if self.model.counter % 1440 == 1200:
            self.goal = self.model.exit
            self.path = []
        # Last class, move towards final scheduled classroom
        elif self.model.counter % 1440 == 940:
            self.goal = self.roster[2]
            self.path = []
        # Second class, move towards second scheduled classroom
        elif self.model.counter % 1440 == 700:
            self.goal = self.roster[1]
            if self.pos != self.goal:
                self.path = []
        # First class, move towards first scheduled classroom.
        elif 700 > self.model.counter % 1440 > 540:
            self.goal = self.roster[0]
            self.path = []

        # While trying to reach goal, interact with neighboring agents
        if self.pos != self.goal:
            if not isinstance(self, wall):
                cellmates = self.model.grid.get_neighbors(self.pos, moore=True)
                newCellmates = []
                for cellmate in cellmates:
                    if not abs(cellmate.pos[0]-self.pos[0]) > 1 and \
                            not abs(cellmate.pos[1]-self.pos[1]) > 1 and \
                            not isinstance(cellmate, wall):
                        newCellmates += [cellmate]

                # behavior based on sociability.
                # move away from agent if low sociability
                if self.sociability == 0 and self.random.random() > 0.3:
                    if len(newCellmates) > 0:
                        other = self.random.choice(newCellmates)
                        # find escape route
                        escape = ((self.pos[0] - other.pos[0]),
                                  (self.pos[1] - other.pos[1]))
                        choice = (escape[0] + self.pos[0], escape[1] +
                                  self.pos[1])
                        if self.model.grid.width > choice[0] > 0 and \
                           self.model.grid.height > choice[1] > 0:
                            if self.model.grid.is_cell_empty(choice):
                                self.model.grid.move_agent(self, choice)
                                return

                # stop if talked to if middle sociability
                if self.sociability == 1 and \
                   self.random.random() > self.talking:
                    for neighbor in newCellmates:
                        if neighbor.sociability == 2:
                            if self.talkedto != neighbor:
                                self.talking *= 1.5
                                self.talkedto = neighbor
                            return
                else:
                    self.talking = 0.1

                # stop to talk if there is a neighbor if high sociability
                if self.sociability == 2 and \
                   self.random.random() > self.talking:
                    if len(newCellmates) > 0 and \
                       self.talkedto != newCellmates[0]:
                        self.talking *= 1.5  # lowers prob to keep talking
                        self.talkedto = newCellmates[0]
                        return
                else:
                    self.talking = 0.1

                # use A* to find best path to goal if goal is present
                if self.path == []:
                    self.path = AStarSearch(self.pos, self.goal, self.model)
                if self.path != []:
                    if self.path != [-1] and \
                       self.model.grid.is_cell_empty(self.path[0]):
                        self.model.grid.move_agent(self, self.path[0])
                        self.path.pop(0)
                    else:
                        self.path = AStarSearch(self.pos, self.goal,
                                                self.model)

    def spread_disease(self):
        """
        Spreads disease to neighbors.
        """
        # Collect cellmates that disease can be spread to, sorted by distance
        cellmates = set(self.model.grid.get_neighbors(self.pos, moore=True))
        cellmates_2 = set(self.model.grid.get_neighbors(self.pos, moore=True,
                                                        radius=2))
        cellmates_3 = set(self.model.grid.get_neighbors(self.pos, moore=True,
                                                        radius=3))
        cellmates_4 = set(self.model.grid.get_neighbors(self.pos, moore=True,
                                                        radius=4))
        cellmates = list(cellmates)
        cellmates_2 = list(cellmates_2.difference(cellmates))
        cellmates_3 = list(cellmates_3.difference(cellmates_2))
        cellmates_4 = list(cellmates_4.difference(cellmates_3))
        # cellmates that are further away have a lower prob of getting sick
        disease_spreader(cellmates, self, 1)
        disease_spreader(cellmates_2, self, 0.75)
        disease_spreader(cellmates_3, self, 0.5)
        disease_spreader(cellmates_4, self, 0.125)

    def mutate(self):
        """
        Mutates disease in an agent.
        """
        # Check if agent has disease to mutate
        if self.disease > 0:
            if self.model.mutateProb > self.random.random():
                self.model.maxDisease += 1
                self.resistant += [self.disease]
                self.disease = self.model.maxDisease
                self.sickTime = 0

    def go_home(self):
        """Remove agent from the grid as they 'go home'."""
        if self.pos == self.model.exit:
            self.model.removed += [self]
            self.model.grid.remove_agent(self)

    def cured(self):
        """
        Cure agents based on cure probability sick time.
        """
        # people are generally sick for at least 1 week (60 * 24 * 7 = 10080)
        if self.sickTime > 10080:
            # Agent is cured
            if self.cureProb > self.random.random():
                self.resistant += [self.disease]
                self.disease = 0
                self.sickTime = 0
                self.cureProb = self.model.initialCureProb
            else:
                self.cureProb *= self.model.cureProbFac

    def step(self):
        """Move and spread disease if sick."""
        # start of schoolday, set agent on grid at entrance (if edu setting)
        if self.model.counter % 1440 > 540 and self.pos is None and \
           self.model.edu_setting is True:
            # place agent if entrance is not occupied by other agent
            if self.model.grid.is_cell_empty(self.model.exit):
                self.model.grid.place_agent(self, self.model.exit)
        # move agent according to setting as long as it is on grid
        if self.pos is not None:
            if self.model.edu_setting is False:
                self.random_move()
            else:
                self.move()
                # end of schoolday, move towards entrance/exit and off the grid
                if self.model.counter % 1440 > 800:
                    self.go_home()
        # if agent carries disease, potentially spread, mutate and/or heal
        if self.disease >= 1:
            self.sickTime += 1
            self.mutate()
            if self.pos is not None:
                self.spread_disease()
            self.cured()
