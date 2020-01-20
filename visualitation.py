from mesa.visualization.modules import CanvasGrid,ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.TextVisualization import TextData
from disease import DiseaseModel
import matplotlib.pyplot as plt
color_array = []
for i in range(100):
	counter = i
	i = i%16
	if i == 10:
		i = 'A'
	if i == 11:
		i = 'B'
	if i == 12:
		i = 'C'
	if i == 13:
		i = 'D'
	if i == 14:
		i = 'E'
	if i == 15:
		i = 'F'
	if i == 16:
		i = 'G'
	if counter%3 == 0:
		color_array += ["#"+"00"+"00"+str(i)+str(i)]
	if counter%3 == 1:
		color_array += ["#"+"00"+str(i)+str(i)+"00"]
	if counter%3 == 2:
		color_array += ["#"+str(i)+str(i)+"00"+"00"]
print(color_array)
color_array = ["blue","red","black"]
def agent_portrayal(agent):
	portrayal = {"Filled": "true","Layer": 2,
						"r": 1}
	if agent.disease > -1:
			portrayal["Shape"] = "circle"
			portrayal["r"] = 1
			portrayal["Color"] = color_array[agent.sociability]
	# portrayal["r"] = 0.5
	else:
		portrayal["Shape"] = "rect"
		portrayal["w"] = 1
		portrayal["h"] = 1
		portrayal["Color"] = "grey"

	return portrayal

# chart = ChartModule([{"Label": "Disease",
# 						"Color": "Black"}],
# 						data_collector_name='datacollector')

grid = CanvasGrid(agent_portrayal, 50, 50, 500, 500)
server = ModularServer(DiseaseModel,
							[grid],
							"Red = sick: Disease Model",
							{"highS":10,"middleS":10,"lowS":10, "width":50, "height":50,"edu_setting":False})

server.port = 8521 # The default
server.launch()
