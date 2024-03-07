import uav_env_v0
# discrete env
dis_env = uav_env_v0.parallel_env(N=3, continuous_actions=False)
# continuous env
con_env = uav_env_v0.parallel_env(N=3, continuous_actions=True)
dis_env.reset()
con_env.reset()
print(dis_env.action_space('agent_0').sample()) # 2
print(con_env.action_space('agent_0').sample()) # array([0.24120373, 0.83279127, 0.4586939 , 0.4208583 , 0.97381055], dtype=float32)
