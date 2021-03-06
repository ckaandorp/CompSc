from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from disease import DiseaseModel
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
from scipy import stats


def t_test(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return "t_value = " + str(round(t, 6)) + "\tp_value = " + str(round(p, 6))


def disease_graph(models, steps, edu_setting):
    """"
    Plots progress of disease given a model.
    """
    diseased_avg = []
    lowS_sick_avg = []
    middleS_sick_avg = []
    highS_sick_avg = []
    lowS_resistant_avg = []
    middleS_resistant_avg = []
    highS_resistant_avg = []
    lowS_avg = []
    middleS_avg = []
    highS_avg = []
    disease_plotter_avg = []
    low_last = []
    mid_last = []
    high_last = []
    max_n_mutations = 0


    for model in models:
        # get dataframe for all timesteps in the model
        df = model.datacollector.get_model_vars_dataframe()
        # initialize store vars
        diseased = []
        mutation = []
        low_sociability = []
        middle_sociability = []
        high_sociability = []
        low_resistant = []
        middle_resistant = []
        high_resistant = []
        n_mutations = 0

        # collect model data
        for index, row in df.iterrows():
            diseased += [row[0][0]]
            mutation += [row[0][1]]
            sociability = row[0][3]
            resistant = row[0][4]
            low_resistant += [resistant['0']]
            middle_resistant += [resistant['1']]
            high_resistant += [resistant['2']]
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
        lowS_sick = [x / model.lowS for x in low_sociability]
        middleS_sick = [x / model.middleS for x in middle_sociability]
        highS_sick = [x / model.highS for x in high_sociability]
        lowS_resistant = [x / model.lowS for x in low_resistant]
        middleS_resistant = [x / model.middleS for x in middle_resistant]
        highS_resistant = [x / model.highS for x in high_resistant]

        # store for averaging
        diseased_avg += [diseased]
        lowS_sick_avg += [lowS_sick]
        middleS_sick_avg += [middleS_sick]
        highS_sick_avg += [highS_sick]
        lowS_resistant_avg += [lowS_resistant]
        middleS_resistant_avg += [middleS_resistant]
        highS_resistant_avg += [highS_resistant]
        lowS_avg += [model.lowS]
        middleS_avg += [model.middleS]
        highS_avg += [model.highS]
        disease_plotter_avg += [disease_plotter]
        low_last += [low_sociability[-1]/model.lowS]
        mid_last += [middle_sociability[-1]/model.middleS]
        high_last += [high_sociability[-1]/model.highS]

    # Write data to textfile
    F = open("workfile.txt", "a")
    F.write("Comparing the means of the percentage of infected agents at")
    F.write(" the last timestep: \n")
    F.write("The current educational setting is " + str(edu_setting) + "\n\n")
    F.write("Percentage infected agents low sociability \t\t" +
            str(np.mean(low_last)) + "\n")
    F.write("Percentage infected agents middle sociability \t" +
            str(np.mean(mid_last)) + "\n")
    F.write("Percentage infected agents high sociability \t" +
            str(np.mean(high_last)) + "\n\n")
    F.write("Low sociability versus middle sociability \t" +
            t_test(low_last, mid_last) + "\n")
    F.write("Low sociability versus high sociability \t" +
            t_test(low_last, high_last) + "\n")
    F.write("Middle sociability versus high sociability \t" +
            t_test(high_last, mid_last) + "\n")
    F.write("----------------------------------------------------------------")
    F.write("------------------------\n\n")

    # Calculate averages
    diseased_avg = np.mean(np.array(diseased_avg), axis=0)

    lowS_sick_avg = np.mean(np.array(lowS_sick_avg), axis=0)
    middleS_sick_avg = np.mean(np.array(middleS_sick_avg), axis=0)
    highS_sick_avg = np.mean(np.array(highS_sick_avg), axis=0)

    lowS_resistant_avg = np.mean(np.array(lowS_resistant_avg), axis=0)
    middleS_resistant_avg = np.mean(np.array(middleS_resistant_avg), axis=0)
    highS_resistant_avg = np.mean(np.array(highS_resistant_avg), axis=0)

    lowS_avg = np.mean(lowS_avg)
    middleS_avg = np.mean(middleS_avg)
    highS_avg = np.mean(highS_avg)

    for mutation_list in disease_plotter_avg:
        len_mutation = len(mutation_list)
        if len_mutation < max_n_mutations:
            mutation_list.extend([[0 for x in range(0, steps)]] *
                                 (max_n_mutations - len_mutation))

    disease_plotter_avg = np.mean(disease_plotter_avg, axis=0)
    plt.plot(diseased_avg, color="red", label='total')

    # Plot all diseases
    for mutation in disease_plotter_avg:
        plt.plot(mutation)
    axes = plt.gca()
    axes.set_ylim([0, 1.1])
    plt.ylabel("Infected (%)")
    plt.xlabel("Timesteps")
    plt.title("Infected agents in " + str(edu_setting) +
              " educational setting")
    axes = plt.gca()
    axes.set_ylim([0, 1.1])
    plt.xlabel('Timesteps')
    plt.ylabel('Infected (%)')
    plt.title("Infected agents in " + str(edu_setting) +
              " educational setting")
    plt.legend()
    plt.show()

    # Plot resistance
    plt.plot(lowS_resistant_avg, label='Low sociability, total agents: '
             + str(int(lowS_avg)))
    plt.plot(middleS_resistant_avg, label='Middle sociability, total agents: '
             + str(int(middleS_avg)))
    plt.plot(highS_resistant_avg, label='High sociability, total agents: '
             + str(int(highS_avg)))
    plt.ylabel("Amount of resistantcy on average per agent")
    plt.xlabel("Timesteps")
    plt.title("Resistance agents in " + str(edu_setting) +
              " educational setting")
    plt.legend()
    plt.show()

    # Write data to textfile
    F = open("workfile.txt", "a")
    F.write("Comparing the means of the average resistance of agents at the")
    F.write(" last timestep: \n")
    F.write("The current educational setting is " + str(edu_setting) + "\n\n")
    F.write("Average resistance of agents low sociability \t\t" +
            str(np.mean(lowS_resistant)) + "\n")
    F.write("Average resistance of agents middle sociability \t" +
            str(np.mean(middleS_resistant)) + "\n")
    F.write("Average resistance of agents high sociability \t\t" +
            str(np.mean(highS_resistant)) + "\n\n")
    F.write("Low sociability versus middle sociability \t" +
            t_test(lowS_resistant, middleS_resistant) + "\n")
    F.write("Low sociability versus high sociability \t" +
            t_test(lowS_resistant, highS_resistant) + "\n")
    F.write("Middle sociability versus high sociability \t" +
            t_test(highS_resistant, middleS_resistant) + "\n")
    F.write("----------------------------------------------------------------")
    F.write("-----------------------\n\n\n")

    # Plot agent sociability
    axes = plt.gca()
    axes.set_ylim([0, 1.1])
    plt.plot(lowS_sick_avg, label='Low sociability, total agents: ' +
             str(int(lowS_avg)))
    plt.plot(middleS_sick_avg, label='Middle sociability, total agents: ' +
             str(int(middleS_avg)))
    plt.plot(highS_sick_avg, label='High sociability, total agents: ' +
             str(int(highS_avg)))
    plt.ylabel("Infected (%)")
    plt.xlabel("Timesteps")
    plt.title("Infected agents in " + str(edu_setting) +
              " educational setting")
    plt.legend()
    plt.show()
    return (np.mean(low_last), np.mean(mid_last), np.mean(high_last))


def graph_edu_non(low_0, mid_0, high_0, low_1, mid_1, high_1, edu_setting):
    # Set width of bar
    barWidth = 0.25
    bars1 = [low_0, low_1]
    bars2 = [mid_0, mid_1]
    bars3 = [high_0, high_1]

    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Set plot dimensions
    axes = plt.gca()
    axes.set_ylim([0, 1])

    # Make the plot
    plt.bar(r1, bars1, width=barWidth, edgecolor='white',
            label='Low sociability')
    plt.bar(r2, bars2, width=barWidth, edgecolor='white',
            label='Middle sociability')
    plt.bar(r3, bars3, width=barWidth, edgecolor='white',
            label='High sociability')

    # Add xticks on the middle of the group bars
    plt.xlabel('Disease rate per sociability in two settings',
               fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))],
               ['Educational setting ' + str(edu_setting),
                'Educational setting ' + str(not edu_setting)])

    # Create legend & Show graphic
    plt.ylabel("Infected (%)")
    plt.title("Infected agents at the last timestep")
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
    # 0th color is always black and healthy
    color_array.insert(0, "#000000")
    return color_array


color_array = color_maker()


def agent_portrayal(agent):
    """Colors and shapes the agent on the grid visualization."""
    portrayal = {"Filled": "true", "Layer": 0, "r": 0.5}
    # draw agent
    if agent.disease > -1:
        portrayal["Shape"] = "circle"
        # change shape if agent has reached its goal
        if agent.goal == agent.pos:
            portrayal["r"] = 2
        else:
            portrayal["r"] = 1
        portrayal["Color"] = color_array[agent.disease % len(color_array)]
    # draw wall
    else:
        portrayal["Shape"] = "rect"
        portrayal["w"] = 1
        portrayal["h"] = 1
        portrayal["Color"] = "grey"
    return portrayal


def visualization_grid(width, height, highS, middleS, lowS, edu_setting=False,
                       cureProb=0.1, cureProbFac=2/1440, mutateProb=0.0050,
                       diseaseRate=0.2):
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
                  diseaseRate=0.2, grid=True, graphs=True, steps=300):
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
        # create an average over different models
        models_0, models_1 = [], []
        for i in range(0, 10):
            model_0 = DiseaseModel(highS, middleS, lowS, width, height,
                                   edu_setting, cureProb, cureProbFac,
                                   mutateProb, diseaseRate)
            model_1 = DiseaseModel(highS, middleS, lowS, width, height,
                                   not edu_setting, cureProb, cureProbFac,
                                   mutateProb, diseaseRate)
            for j in range(steps):
                # print every 100th step to inform user about progress
                if j % 100 == 0:
                    print(j)
                model_0.step()
                model_1.step()

            models_0 += [model_0]
            models_1 += [model_1]

        low_0, mid_0, high_0 = disease_graph(models_0, steps, edu_setting)
        low_1, mid_1, high_1 = disease_graph(models_1, steps, not edu_setting)
        graph_edu_non(low_0, mid_0, high_0, low_1, mid_1, high_1, edu_setting)

    # visualization on server
    if grid:
        visualization_grid(width, height, highS, middleS, lowS, edu_setting,
                           cureProb, cureProbFac, mutateProb, diseaseRate)


# Run shorter version for demo if d(emo) flag is set
if len(sys.argv) == 2 and sys.argv[1] == "-d":
    F = open("workfile.txt", "w")
    F.write("")
    visualization(50, 50, 10, 10, 10, steps=100, grid=True, edu_setting=True)
else:
    F = open("workfile.txt", "w")
    F.write("")
    visualization(50, 50, 10, 10, 10, steps=30000, edu_setting=False,
                  cureProb=0.2, cureProbFac=2/1440, mutateProb=0.0000050,
                  diseaseRate=0.02, grid=False, graphs=True)
