# CompSc
Simulation of the spread of the common cold when looking at social interactions. <br>

This repostory consists of the following files, <br>
disease.py, Contains the disease model. <br>
diseaseAgent.py, Contains the agent model. <br>
helperFunctions.py, Contains Function to help with the running of the models.<br>
visualization.py, Contains visualization of the model. <br>
wall.py, Contains agent wall model. <br>

To run our model you can run visualization.py with the flag -d. 
That is `python3 visualization.py -d`, this will first run a demo version of the model for graph plotting.
Then it will run the model streaming wise with a moving visualization in your browser. <br>
If your browser doesnt open automatically, run the code and then go to: http://127.0.0.1:8521.
Our dependencies are mesa, numpy, matplotlib.

So please run: <br>
`pip install -r requirements.txt`<br>
before runnig our code.

You should get comparable graphs to these when running the demo. 
The values of your workfile.txt should be comparable to the one in the main folder.
But please note they will be slightly different since there is randomness involved in our model.
![infect_nonedu_mutations](/Graphs/infect_nonedu_mutations.png)
![infect_nonedu_social](/Graphs/infect_nonedu_social.png)
![infect_res_nonedu](/Graphs/res_nonedu.png)
![infect_edu_mutations](/Graphs/infect_edu_mutations.png)
![infect_edu_social](/Graphs/infect_edu_social.png)
![res_edu](/Graphs/res_edu.png)
![infect_bar](/Graphs/infect_bar.png)
<br>
The full version can be run with `python3 visualization.py` and will start a simulation spanning approximately 3 weeks. As this simulation will take a long time to run it is not recommended for the code review. <br>
<br>
If you want to change the paramaters your self. You can do so in visualization.py. By running the function visualization(). The optional parameters are: <br>
<!---
width = width of the grid                                                                         (int) <br>
height = height of the grid                                                                       (int) <br>
highS = number of agents with high sociability                                                    (int) <br>
middleS = number of agents with middle sociability                                                (int) <br>
lowS = number of agents with low sociability                                                      (int) <br>
edu_setting(True) default, setting between educational or random                                  (bool) <br>
cureProb(0.1) default, probability of getting better after 1 week time                            (float) <br>
cureProbFac(2/1440) default, factor by which your cure prob is increased every step after 1 week  (float) <br>
mutateProb(0.0050) default, probability of a mutation occuring                                    (float) <br>
diseaseRate(0.2) default, rate at wich the dissease spreads                                       (float) <br>
grid(True) default, option if you want live visualization in browser                              (bool) <br>
graphs(True) default, option if you want graphs visualization                                     (bool) <br>
steps(300) default, amount of steps the models take.                                              (int) <br>
--->

Parameter | Default | Description | Type
--- | --- | --- | ---
width | |width of the grid|                                                                       (int) 
height | | height of the grid |                                                                      (int) 
highS | | number of agents with high sociability |                                                    (int) 
middleS | | number of agents with middle sociability |                                               (int) 
lowS | | number of agents with low sociability       |                                               (int) 
edu_setting|True| setting between educational or random  |                                (bool) 
cureProb|0.1|probability of getting better after 1 week time|                            (float)
cureProbFac|2/1440| factor by which your cure prob is increased every step after 1 week | (float)
mutateProb|0.0050| probability of a mutation occuring                                    |(float)
diseaseRate|0.2|rate at wich the dissease spreads                                       |(float)
grid|True| option if you want live visualization in browser                              |(bool) 
graphs|True| option if you want graphs visualization                                     |(bool)
steps |300| amount of steps the models take.                                              |(int) 
