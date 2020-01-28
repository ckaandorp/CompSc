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
But please note they will be slightly different since there is randomness involved in our model.
![infect_nonedu_mutations](/Graphs/infect_nonedu_mutations.png)
![infect_nonedu_social](/Graphs/infect_nonedu_social.png)
![infect_res_nonedu](/Graphs/res_nonedu.png)
![infect_edu_mutations](/Graphs/infect_edu_mutations.png)
![infect_edu_social](/Graphs/infect_edu_social.png)
![res_edu](/Graphs/res_edu.png)
![infect_bar](/Graphs/infect_bar.png)

The full version can be run with `python3 visualization.py` and will start a simulation spanning approximately 3 weeks. As this simulation will take a long time to run it is not recommended for the code review.


