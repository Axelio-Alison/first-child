# First Child Project: Deep Q-Learning for Portfolio Management
Hi folks, this is my first serious (and a little bit complex) project about __Reinforcement Learning__ for financial applications. Moreover, this is also a part of my Bachelor's Degree final thesis on _"Forecasting Models and Deep Q-Learning for Portfolio Management"_, a thesis in which I combine forecasting models and Reinforcement Learning techniques to create an Asset Management bot.

## Structure of the Project
The principal file on which you will find the results of my model are in the [first_child.ipynb](https://github.com/Axelio-Alison/first-child/blob/main/first_child.ipynb) file, while the files in the __Python__ directory contained functional modules for defining a custom _Trading Environment_ for portfolio management and creating a custom _D3QN Model_. 

In the Jupyter Notebook file you can find the project that contains _original_ images useful to visualize and explain the work. 
After the problem definition section, the file is divided in two sections:
1. __Data Analysis and Benchmark Creation:__ the secton in which I select the assets to use, analyse the correlation between the assets and their past returns...

<!-- [<img src="Assets Correlation Matrix.png" width="20"/>](https://github.com/user-attachments/assets/7b9f868d-7b77-45f2-bb1f-a4d99fa38aa9)
<p align="center">
  <img src="https://github.com/user-attachments/assets/7b9f868d-7b77-45f2-bb1f-a4d99fa38aa9" alt = "Assets Correlation Matrix" width="600px">
</p> -->

<p align="center">
  <img src="https://github.com/user-attachments/assets/1d4f81eb-4c94-413b-ab9b-0c0870707ef2" alt = "Assets Return Charts" width="700px">
</p>

... and select the two benchmarks to compare to agent results using a Montecarlo simulation.

<p align = "center">
  <img src = "https://github.com/user-attachments/assets/d51affdc-9709-4d85-9057-8d531a59adec" alt = "Montecarlo Simulation and Efficient Frontier" width = "700px">
</p>
