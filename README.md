![Static Badge](https://img.shields.io/badge/python-3.9.7-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54&link=https%253A%252F%252Fimg.shields.io) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

# First Child Project: Deep Q-Learning for Portfolio Optimization
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
  <img src = "https://github.com/user-attachments/assets/d51affdc-9709-4d85-9057-8d531a59adec" alt = "Montecarlo Simulation and Efficient Frontier" width = "600px">
</p>

2. __Agent Training:__ the section in which I preprocess input features, train the model and expose the results of my study.

<p align = "center">
 <img src = "https://github.com/user-attachments/assets/3fedb029-0266-40e2-ba8c-fe4acefaa5fe" alt = "New Efficient Frontier" width = "600px">
</p>

## Conclusion
To give a slight spoiler, the model has generated promising results, but additional tests need to be done to verify its effectiveness. Moreover, I am working to implement the part on traditional forecasting models in the code to improve the model's metrics.

