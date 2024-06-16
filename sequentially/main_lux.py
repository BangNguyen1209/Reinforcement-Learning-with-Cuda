#from stable_baselines3 import PPO  # pip install stable-baselines3
from Algorythm.PPO import PPO
from lux.env.agent import AgentFromStdInOut
from lux.env.lux_env import LuxEnvironment
from lux.game.constants import LuxMatchConfigs_Default
from agent_personal_custom import AgentPersonalCustom

if __name__ == "__main__":
    # Run a kaggle submission with the specified model
    configs = LuxMatchConfigs_Default

    # Load the saved model
    model = PPO.load(f"model.zip")
    
    # Create a kaggle-remote opponent agent
    opponent = AgentFromStdInOut()

    # Create a RL agent in inference mode
    player = AgentPersonalCustom(mode="inference", model=model)

    # Run the environment
    env = LuxEnvironment(configs, player, opponent)
    env.reset()  # This will automatically run the game since there is
    # no controlling learning agent.
