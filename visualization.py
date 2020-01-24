from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from disease import DiseaseModel
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import stats

def t_test(a, b):
	t, p = stats.ttest_ind(a, b)
	print('t', t)
	print('p', p)
	return "t_value = " + str(round(t,6)) +"  p_value = " + str(round(p,6))

def disease_graph(models, steps):
	""""
	Plots progress of disease given a model.
	"""
	diseased_avg = []
	lowS_sick_avg = []
	middleS_sick_avg = []
	highS_sick_avg = []
	lowS_resistent_avg = []
	middleS_resistent_avg = []
	highS_resistent_avg = []
	lowS_avg = []
	middleS_avg = []
	highS_avg = []
	disease_plotter_avg = []
	low_last = []
	mid_last = []
	high_last = []
	max_n_mutations = 0

	for model in models:
		# get dataframe
		df = model.datacollector.get_model_vars_dataframe()

		# initialize store vars
		diseased = []
		mutation = []
		low_sociability = []
		middle_sociability = []
		high_sociability = []
		low_resistent = []
		middle_resistent = []
		high_resistent = []
		n_mutations = 0


		for index, row in df.iterrows():
			diseased += [row[0][0]]
			mutation += [row[0][1]]
			sociability = row[0][3]
			resistent = row[0][4]
			low_resistent += [resistent['0']]
			middle_resistent += [resistent['1']]
			high_resistent += [resistent['2']]
			low_sociability += [sociability['0']]
			middle_sociability += [sociability['1']]
			high_sociability += [sociability['2']]
			if row[0][2] > n_mutations:
				n_mutations = row[0][2]
				if n_mutations > max_n_mutations:
					max_n_mutations = n_mutations


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
			# print("HELP\n", disease_plotter)
			# print()
		print("mutationlist", disease_plotter[0])
		print()

		lowS_sick = [x / model.lowS for x in low_sociability]
		middleS_sick = [x / model.middleS for x in middle_sociability]
		highS_sick = [x / model.highS for x in high_sociability]

		lowS_resistent = [x / model.lowS for x in low_resistent]
		middleS_resistent = [x / model.middleS for x in middle_resistent]
		highS_resistent = [x / model.highS for x in high_resistent]
		# store for averaging
		diseased_avg += [diseased]

		lowS_sick_avg += [lowS_sick]
		middleS_sick_avg += [middleS_sick]
		highS_sick_avg += [highS_sick]

		lowS_resistent_avg += [lowS_resistent]
		middleS_resistent_avg += [middleS_resistent]
		highS_resistent_avg += [highS_resistent]

		lowS_avg += [model.lowS]
		middleS_avg += [model.middleS]
		highS_avg += [model.highS]
		disease_plotter_avg += [disease_plotter]

		low_last += [low_sociability[-1]/model.lowS]
		mid_last += [middle_sociability[-1]/model.middleS]
		high_last += [high_sociability[-1]/model.highS]
		print(low_last)
		print(mid_last)
		print(high_last)

	F = open("workfile.txt","w")
	F.write("low sociability versus middle sociability  " + t_test(low_last,mid_last) + "\n")
	F.write("low sociability versus high sociability    " + t_test(low_last,high_last)+ "\n")
	F.write("middle sociability versus high sociability " + t_test(high_last,mid_last)+ "\n")
	### calculate averages + plot
	diseased_avg = np.mean(np.array(diseased_avg), axis=0)
	print()
	print("AVERAGE\n", disease_plotter_avg[0])
	print()
	lowS_sick_avg = np.mean(np.array(lowS_sick_avg), axis=0)
	middleS_sick_avg = np.mean(np.array(middleS_sick_avg), axis=0)
	highS_sick_avg = np.mean(np.array(highS_sick_avg), axis=0)

	lowS_resistent_avg = np.mean(np.array(lowS_resistent_avg), axis=0)
	middleS_resistent_avg = np.mean(np.array(middleS_resistent_avg), axis=0)
	highS_resistent_avg = np.mean(np.array(highS_resistent_avg), axis=0)

	lowS_avg = np.mean(lowS_avg)
	middleS_avg = np.mean(middleS_avg)
	highS_avg = np.mean(highS_avg)

	for mutation_list in disease_plotter_avg:
		len_mutation = len(mutation_list)
		if len_mutation < max_n_mutations:
			mutation_list.extend([[0 for x in range(0, steps)]] * (max_n_mutations - len_mutation))

	disease_plotter_avg = np.mean(disease_plotter_avg, axis=0)
	plt.plot(diseased_avg, color="red", label='total')
	axes = plt.gca()
	axes.set_ylim([0, 1])
	plt.xlabel('Timesteps')
	plt.ylabel('Infected (%)')
	plt.legend()
	plt.show()

	print(lowS_resistent_avg)
	plt.plot(lowS_resistent_avg, label='Low sociability, total agents: ' + str(int(lowS_avg)))
	plt.plot(middleS_resistent_avg,label='Middle sociability, total agents: ' + str(int(middleS_avg)))
	plt.plot(highS_resistent_avg, label='High sociability, total agents: ' + str(int(highS_avg)) )
	plt.ylabel("Amount of resistentcy on average per agent")
	plt.xlabel("Timesteps")
	plt.legend()
	plt.show()
	# plot all diseases
	for mutation in disease_plotter_avg:
		plt.plot(mutation)



	# plot agent sociability
	axes = plt.gca()
	axes.set_ylim([0, 1])
	plt.plot(lowS_sick_avg, label='Low sociability, total agents: ' + str(int(lowS_avg)))
	plt.plot(middleS_sick_avg, label='Middle sociability, total agents: ' + str(int(middleS_avg)))
	plt.plot(highS_sick_avg, label='High sociability, total agents: ' + str(int(highS_avg)))
	plt.ylabel("Infected (%)")
	plt.xlabel("Timesteps")
	plt.legend()
	plt.show()
	return (np.mean(low_last),np.mean(mid_last),np.mean(high_last))
def graph_edu_non(low,mid,high,edlow,edmid,edhigh):
	# set width of bar
	barWidth = 0.25
	bars1 = [edlow, low]
	bars2 = [edmid, mid]
	bars3 = [edhigh,high]

	# Set position of bar on X axis
	r1 = np.arange(len(bars1))
	r2 = [x + barWidth for x in r1]
	r3 = [x + barWidth for x in r2]

	# Make the plot
	plt.bar(r1, bars1,width=barWidth, edgecolor='white', label='Low sociability')
	plt.bar(r2, bars2,width=barWidth, edgecolor='white', label='Middle sociability')
	plt.bar(r3, bars3,width=barWidth, edgecolor='white', label='High sociability')

	axes = plt.gca()
	axes.set_ylim([0, 1])
	# Add xticks on the middle of the group bars
	plt.xlabel('Disease rate per sociability in two settings.', fontweight='bold')
	plt.xticks([r + barWidth for r in range(len(bars1))], ['Educational', 'Random movement'])

	# Create legend & Show graphic
	plt.legend()
	plt.show()

def color_maker():
	"""Returns a list of colors."""
	R, G, B = 0, 0, 0
	color_array = []

	for i in range(1, 6):
		for j in range(1, 6):
			for k in range(1, 6):
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


color_array = color_maker()


def agent_portrayal(agent):
	portrayal = {"Filled": "true", "Layer": 0, "r": 0.5}
	if agent.disease > -1:
		portrayal["Shape"] = "circle"
		if agent.goal == agent.pos:
			portrayal["r"] = 2
		else:
			portrayal["r"] = 1
		portrayal["Color"] = color_array[agent.disease % len(color_array)]
	else:
		portrayal["Shape"] = "rect"
		portrayal["w"] = 1
		portrayal["h"] = 1
		portrayal["Color"] = "grey"
	return portrayal


def visualization_grid(width, height, highS, middleS, lowS, edu_setting=False,
						cureProb=0.1, cureProbFac=2/1440, mutateProb=0.0050,
						diseaseRate=0.38):
	"""
	Launch grid visualization on server.
	width: Width of the grid.
	height: Height of the grid.
	highS: Number of agents with high sociability.
	middleS: Number of agents with middle sociability.
	lowS: Number of agents with low sociability.
	edu_setting: If true, agents will follow a schedule and sit in classrooms,
	else they will move freely through an open grid.
	cureProb: Probability of agent getting better.
	cureProbFac: Factor of cureProb getting higher.
	mutateProb: Probability of a disease mutating.
	diseaseRate: Rate at which the disease spreads
	"""
	grid = CanvasGrid(agent_portrayal, width, height, width*10, height*10)
	server = ModularServer(DiseaseModel, [grid], "Disease Model",
							{"highS": highS, "middleS": middleS, "lowS": lowS,
							"width": width, "height": height,
							"edu_setting": edu_setting, "cureProb": cureProb,
							"cureProbFac": cureProbFac,
							"mutateProb": mutateProb,
							"diseaseRate": diseaseRate})
	server.port = 8521   # The default
	server.launch()


def visualization(width, height, highS, middleS, lowS, edu_setting=True,
					cureProb=0.1, cureProbFac=2/1440, mutateProb=0.0050,
					diseaseRate=0.38, grid=True, graphs=True, steps=300):
	"""
	Create visualizations.
	width: Width of the grid.
	height: Height of the grid.
	highS: Number of agents with high sociability.
	middleS: Number of agents with middle sociability.
	lowS: Number of agents with low sociability.
	edu_setting: If true, agents will follow a schedule and sit in classrooms,
	else they will move freely through an open grid.
	cureProb: Probability of agent getting better.
	cureProbFac: Factor of cureProb getting higher.
	mutateProb: Probability of a disease mutating.
	diseaseRate: Rate at which the disease spreads.
	grid: if True show grid visualisation.
	graphs: if True show graphs.
	steps: number of steps in graph.
	"""
	if graphs:
		# create an average
		models = []
		for i in range(0, 10):
			model = DiseaseModel(highS, middleS, lowS, width, height,
								edu_setting, cureProb, cureProbFac,
								mutateProb, diseaseRate)
			for j in range(steps):
				print(j)
				model.step()
			models += [model]

	low,mid,high = disease_graph(models, steps)

	if grid:
		visualization_grid(width, height, highS, middleS, lowS, edu_setting,
							cureProb, cureProbFac, mutateProb, diseaseRate)
	return low,mid,high

low,mid,high = visualization(50, 50, 10, 10, 10, steps=10,grid= False,edu_setting=False)
eduLow,eduMid,eduHigh = visualization(50, 50, 10, 10, 10, steps=10,grid = False)
graph_edu_non(low,mid,high,eduLow,eduMid,eduHigh)
