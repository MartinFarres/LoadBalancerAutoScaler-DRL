from stable_baselines3.common.env_checker import check_env
from environment import LoadBalancerEnv

env = LoadBalancerEnv(n_max=10, max_steps=100, max_memory=1024)

print("Running Environment Checker...")
check_env(env)
print("Passed all env checks!")
print("Checking random action sample interaction...")
state, info = env.reset()
for i in range(50):
    sample = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(sample)

    print("Reward: ", reward)
    print("Info: ", info)

    if terminated or truncated:
        i = 50
        print("Terminating...")
print("Random actions sample Passed")
