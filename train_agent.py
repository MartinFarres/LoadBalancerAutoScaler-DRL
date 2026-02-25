from stable_baselines3 import PPO
from environment import LoadBalancerEnv 

env = LoadBalancerEnv() 

# "MlpPolicy" -> Multi Layer Perceptron
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

print("Initiating training sequence...")

model.learn(total_timesteps=1000000) # 1M

model.save("ppo_lb_agent_v1")
print("Training complete and model saved!")