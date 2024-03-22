import numpy as np
# 取消随机化任务
# D_m = np.random.randint(1, 5)  # Mbits size_of_input_data
# C_m = np.random.randint(100, 200)  # cycles/bit number_of_cpu_cycles
D_m = 1  # Mbits size_of_input_data
C_m = 100  # cycles/bit number_of_cpu_cycles
Lambda_m = 1  # tasks/second arrival_rate_of_tasks
L_max_h = 49  # meters max_horizontal_distance
D_min = 100  # meters min_distance_of_uavs
Phi_n = 42.44  # degrees elevation_angle
g_0 = -50  # dB path_loss_exponent
B_u = 10  # MHz uplink_channel_bandwidth
B_k = 0.5  # MHz bandwidth_preassigned_to_ecs
P_max = 5  # W max_transmit_power_of_uavs
P_m = 0.1  # W transmit_power_of_ue
P_n_r = 0.1  # W receiving_power_of_uavs
F_u = 3  # GHz computation_resource_F_u
# F_e_k = np.random.randint(6, 9)  # GHz computation_resource_F_ek
F_e_k = 6
Kappa = 1e-28  # effective_switched_capacitance
Sigma_u_2 = -100  # dBm noise_power_sigma_u2
Sigma_e_2 = -100  # dBm noise_power_sigma_e2
w1 = 1  # weights_w1
w2 = 0.001  # weights_w2
Eta1 = 5  # penalty_cofficient_of_uavs_overlapping_n1
Eta2 = 5  # penalty_of_uavs_collision_n2
Eta3 = 5  # penalty_of_uavs_coverage_n3
