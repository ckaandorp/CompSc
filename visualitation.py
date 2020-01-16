from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from disease import DiseaseModel

color_array = ["#0000FE","#FF00FA","#00FFF5","#0000F0","#0000EE","#0000EA","#0000E5","#0000E0"]
def agent_portrayal(agent):
	portrayal = {"Filled": "true","Layer": 0,
						"r": 0.5}
	if agent.disease > -1:
			portrayal["Shape"] = "circle"
			portrayal["r"] = 1
			portrayal["Color"] = color_array[agent.disease]
	# portrayal["r"] = 0.5
	else:
		portrayal["Shape"] = "rect"
		portrayal["w"] = 1
		portrayal["h"] = 1
		portrayal["Color"] = "grey"
	return portrayal

grid = CanvasGrid(agent_portrayal, 50, 50, 500, 500)
server = ModularServer(DiseaseModel,
							[grid],
							"Disease Model",
							{"highS":10,"middleS":10,"lowS":10, "width":50, "height":50,"edu_setting":True})
server.port = 8521 # The default
server.launch()
