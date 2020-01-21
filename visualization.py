from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from disease import DiseaseModel
import matplotlib.pyplot as plt
import random


def disease_graph(model):
	""""
	Plots progress of disease given a model.
	"""
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

def color_maker():
	"""Returns a list of colors."""
	R, G, B = 0, 0, 0
	color_array = []

	for i in range(1,6):
		for j in range(1,6):
			for k in range(1,6):
				# skip gray
				if i == j == k:
					continue
				R = i * 50
				G = j * 50
				B = k * 50
				# Ensure colors are not too dark
				if (R + G + B) > 200:
					color_array += ["#" + hex(R)[2:] + hex(G)[2:] + hex(B)[2:]]
	
	random.shuffle(color_array)
	color_array.insert(0, "#000000")
	return color_array

def agent_portrayal(agent):
	portrayal = {"Filled": "true","Layer": 0,
						"r": 0.5}
	if agent.disease > -1:
			portrayal["Shape"] = "circle"
			portrayal["r"] = 1
			portrayal["Color"] = color_array[agent.disease % len(color_array)]
	else:
		portrayal["Shape"] = "rect"
		portrayal["w"] = 1
		portrayal["h"] = 1
		portrayal["Color"] = "grey"
	return portrayal

color_array = color_maker()

def visualization_grid(width, height, highS, middleS, lowS, edu_setting=False, cureProb=0.1, cureProbFac=2/1440, mutateProb=0.0050, diseaseRate=0.38):
	"""
	Launch grid visualization on server.
	"""
	grid = CanvasGrid(agent_portrayal, width, height, 500, 500)
	server = ModularServer(DiseaseModel,
								[grid],
								"Disease Model",
								{"highS":highS, "middleS":middleS, "lowS":lowS, "width":width, "height":height, "edu_setting":True, "cureProb":0.1, "cureProbFac":2/1440, "mutateProb":0.0050, "diseaseRate":0.38})
	server.port = 8521 # The default
	server.launch()

def visualization(width, height, highS, middleS, lowS, edu_setting=True, cureProb=0.1, cureProbFac=2/1440, mutateProb=0.0050, diseaseRate=0.38, grid=True, graphs=True, steps=300):
	"""
	Create visualizations.
	"""
	if graphs:
		model = DiseaseModel(highS, middleS, lowS, width, height, edu_setting, cureProb, cureProbFac, mutateProb, diseaseRate)
		for i in range(steps):
			print(i)
			model.step()
		disease_graph(model)
	if grid:
		visualization_grid(width, height, highS, middleS, lowS, edu_setting, cureProb, cureProbFac, mutateProb, diseaseRate)

visualization(50, 50, 20, 20, 20)
