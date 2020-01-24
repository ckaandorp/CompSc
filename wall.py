from mesa import Agent


class wall(Agent):
    """A wall seperating the spaces."""
    def __init__(self, unique_id, model):
        self.disease = -1
        super().__init__(unique_id, model)
