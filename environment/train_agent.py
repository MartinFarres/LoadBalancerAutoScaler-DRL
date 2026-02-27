from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from environment import LoadBalancerEnv 

env = Monitor(LoadBalancerEnv())

directory_logs = "./logs_tensorboard/"

# "MlpPolicy" -> Multi Layer Perceptron
model = PPO("MlpPolicy", 
            env, 
            verbose=1, 
            learning_rate=0.0003, 
            tensorboard_log=directory_logs)

print("Initiating training sequence...")

model.learn(total_timesteps=5000, tb_log_name="PPRO_500k") # 5k

model.save("ppo_lb_agent_v1")
print("Training complete and model saved!")