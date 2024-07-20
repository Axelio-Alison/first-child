import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib
import discretization

class PortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, n_assets, X, y, capital: int = 100000):
        super(PortfolioEnv, self).__init__()
        
        # The assets will be the asset selected + cash option
        discretization_r = discretization.Action_discretization(n_assets + 1, 10)
        self.actions_dict = discretization_r[1]
        self.action_n = discretization_r[0]

        self.data = X                                   # Data like pct differences and indicators
        self.returns = y                                # Returns of the assets
        self.returns = self.returns.assign(CASH = 0)    # Add returns of cash
        self.np_random = 10
        self.n_assets = n_assets + 1

        # Set initial weights
        self.initial_weights = np.ones(self.n_assets) / self.n_assets
        self.current_weights = self.initial_weights

        self.starting_capital = capital
        self.capital = capital
        self.capital_x = [self.returns.index[0]]                # Capital History - Date
        self.capital_y = [self.capital]                         # Capital History - Value
        self.capital_weights = [self.current_weights]
        

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(self.action_n)
        self.observation_space = gym.spaces.Box(low = -np.inf, high = np.inf, shape = (len(self.data.columns) + self.n_assets, ), dtype=np.float32)

        self.current_step = 0
        self.max_steps = len(self.data) - 1
    
    def get_actions_dict(self):
        return self.actions_dict

    def _next_observation(self):
    
        # Calculate percentuale variation in prices (this code is usefull only if you take adjusted close as input)
        obs = np.concatenate((self.data.iloc[self.current_step].values, self.current_weights))

        return obs

    # Check this function
    def reset(self):

        self.capital_x = []
        self.capital_y = []

        self.current_step = 0
        self.current_weights = self.initial_weights
        self.capital = self.starting_capital

        return self._next_observation()
    
    def step(self, action: list, render = True):

        self.current_step += 1

        self.current_weights = self.actions_dict[action]

        reward = (self.returns.iloc[self.current_step-1]*self.current_weights).sum()

        self.capital = self.capital * (1 + reward)
        self.capital_x.append(self.returns.index[self.current_step-1])
        self.capital_y.append(self.capital)
        self.capital_weights.append(self.current_weights)

        done = self.current_step >= self.max_steps

        obs = self._next_observation()

        if render:
            self.render()

        return obs, reward, done, {}
    
    def render(self, mode = 'human'):
        
        #print(self.chart, self.ax, self.fig)
        #self.update_chart()
        # if self.current_step == 0:

        #     #self.chart = plt.plot([0], [0])[0]
        #     self.ax.clear()
        
        # if self.current_step > 0:

        #     self.ax.clear()
        #     self.ax.plot(self.capital_x, self.capital_y)
        #     self.fig.canvas.draw()
        #     self.ax.set_xlabel("Date")
        #     self.ax.set_ylabel("Capital Value")

        #     plt.show()

        print(f"\r{self.current_step}/{self.max_steps}\t Portfolio Value: {self.capital:.2f}", end = "")
    
    def update_chart(self):

        pass

    def close(self):
        pass