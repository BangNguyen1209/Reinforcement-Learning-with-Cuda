<<<<<<< HEAD
# Reinforcement-Learning-with-Cuda
=======
# Lux python game engine and gym
This is a replica of the Lux game ported directly over to python. It also sets up a classic Reinforcement Learning gym environment to be used to train RL agents for creating agents.


| **Features**                         | **Check** |
| ------------------------------------ | ----------------------|
| Lux game engine porting to python    | :heavy_check_mark: |
| Documentation                        | :x: |
| Execute the program sequentially | :heavy_check_mark: |
|  | :x:  |
|  | :x: |
|  | :x: |
|  | :x: |
|  | :x: |

# Installation
This should work cross-platform, but I've only tested Windows 10 and Ubuntu.

**Important:** Use Python 3.7.* for training your models. This is required since when you create a Kaggle submission, the Kaggle competition will run the code using Python 3.7.*, and you will get a model deserialization error if you train the model with Python 3.8>=.

Install lux environment package by running the installer:

```python setup.py install```

You will need Node.js version 12 or above: [here](https://nodejs.org/en/download/)


# Python game interface
To directly use the ported game engine without the RL gym wrapper, here a couple example usages:

```
from lux.game.game import Game
from lux.game.actions import *
from lux.game.constants import LuxMatchConfigs_Default


if __name__ == "__main__":
    # Create a game
    configs = LuxMatchConfigs_Default
    game = Game(configs)
    
    game_over = False
    while not game_over:
        print("Turn %i" % game.state["turn"])

        # Array of actions for both teams. Eg: MoveAction(team, unit_id, direction)
        actions = [] 

        game_over = game.run_turn_with_actions(actions)
    
    print("Game done, final map:")
    print(game.map.get_map_string())
```


# Python gym environment interface for RL

A gym interface and match controller was created that supports creating custom agents, and a framework to submit them in kaggle submissions. Keep in mind that this framework is built around one action per unit + city_tile that can act each turn. Creating a basic gym interface looks like the following, however you should look at the more complete example in the examples subfolder:

```
import random
from stable_baselines3 import PPO  # pip install stable-baselines3
from lux.env.lux_env import LuxEnvironment, SaveReplayAndModelCallback
from lux.env.agent import Agent, AgentWithModel
from lux.game.game import Game
from lux.game.actions import *
from lux.game.constants import LuxMatchConfigs_Default
from functools import partial  # pip install functools
import numpy as np
from gym import spaces
import time
import sys

class MyCustomAgent(AgentWithModel):
    def __init__(self, mode="train", model=None) -> None:
        """
        Implements an agent opponent
        """
        super().__init__(mode, model)
        
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.actions_units = [
            partial(MoveAction, direction=Constants.DIRECTIONS.CENTER),  # This is the do-nothing action
            partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.WEST),
            partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.EAST),
            SpawnCityAction,
        ]
        self.actions_cities = [
            SpawnWorkerAction,
            SpawnCartAction,
            ResearchAction,
        ]
        self.action_space = spaces.Discrete(max(len(self.actions_units), len(self.actions_cities)))
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,1), dtype=np.float16)

    def game_start(self, game):
        """
        This function is called at the start of each game. Use this to
        reset and initialize per game. Note that self.team may have
        been changed since last game. The game map has been created
        and starting units placed.

        Args:
            game ([type]): Game.
        """
        pass

    def turn_heurstics(self, game, is_first_turn):
        """
        This is called pre-observation actions to allow for hardcoded heuristics
        to control a subset of units. Any unit or city that gets an action from this
        callback, will not create an observation+action.

        Args:
            game ([type]): Game in progress
            is_first_turn (bool): True if it's the first turn of a game.
        """
        return
    
    def get_observation(self, game, unit, city_tile, team, is_new_turn):
        """
        Implements getting a observation from the current game for this unit or city
        """
        return np.zeros((10,1))
    
    def action_code_to_action(self, action_code, game, unit=None, city_tile=None, team=None):
        """
        Takes an action in the environment according to actionCode:
            action_code: Index of action to take into the action array.
        Returns: An action.
        """
        # Map action_code index into to a constructed Action object
        try:
            x = None
            y = None
            if city_tile is not None:
                x = city_tile.pos.x
                y = city_tile.pos.y
            elif unit is not None:
                x = unit.pos.x
                y = unit.pos.y
            
            if city_tile != None:
                action =  self.actions_cities[action_code%len(self.actions_cities)](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y
                )
            else:
                action =  self.actions_units[action_code%len(self.actions_units)](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y
                )
            
            return action
        except Exception as e:
            # Not a valid action
            print(e)
            return None
    
    def take_action(self, action_code, game, unit=None, city_tile=None, team=None):
        """
        Takes an action in the environment according to actionCode:
            actionCode: Index of action to take into the action array.
        """
        action = self.action_code_to_action(action_code, game, unit, city_tile, team)
        self.match_controller.take_action(action)
    
    def game_start(self, game):
        """
        This function is called at the start of each game. Use this to
        reset and initialize per game. Note that self.team may have
        been changed since last game. The game map has been created
        and starting units placed.

        Args:
            game ([type]): Game.
        """
        pass
    
    def get_reward(self, game, is_game_finished, is_new_turn, is_game_error):
        """
        Returns the reward function for this step of the game. Reward should be a
        delta increment to the reward, not the total current reward.
        """
        if is_game_finished:
            if game.get_winning_team() == self.team:
                return 1 # Win!
            else:
                return -1 # Loss

        return 0
    

if __name__ == "__main__":
    # Create the two agents that will play eachother
    
    # Create a default opponent agent that does nothing
    opponent = Agent()
    
    # Create a RL agent in training mode
    player = MyCustomAgent(mode="train")
    
    # Create a game environment
    configs = LuxMatchConfigs_Default
    env = LuxEnvironment(configs=configs,
                     learning_agent=player,
                     opponent_agent=opponent)
    
    # Play 5 games
    env.reset()
    obs = env.reset()
    game_count = 0
    while game_count < 5:
        # Take a random action
        action_code = random.sample(range(player.action_space.n), 1)[0]
        (obs, reward, is_game_over, state) = env.step( action_code )
        
        if is_game_over:
            print(f"Game done turn {env.game.state['turn']}, final map:")
            print(env.game.map.get_map_string())
            obs = env.reset()
            game_count += 1
    
    # Attach a ML model from stable_baselines3 and train a RL model
    model = PPO("MlpPolicy",
                    env,
                    verbose=1,
                    tensorboard_log="./lux_tensorboard/",
                    learning_rate=0.001,
                    gamma=0.998,
                    gae_lambda=0.95,
                    batch_size=2048,
                    n_steps=2048
                )
    
    print("Training model for 100K steps...")
    model.learn(total_timesteps=10000000)
    model.save(path='model.zip')

    # Inference the agent for 5 games
    game_count = 0
    obs = env.reset()
    while game_count < 5:
        action_code, _states = model.predict(obs, deterministic=False)
        (obs, reward, is_game_over, state) = env.step( action_code )
        
        if is_game_over:
            print(f"Game done turn {env.game.state['turn']}, final map:")
            print(env.game.map.get_map_string())
            obs = env.reset()
            game_count += 1



```
>>>>>>> master
