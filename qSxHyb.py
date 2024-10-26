import numpy as np
import dimod
from dwave.system import LeapHybridCQMSampler
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Set the seed for reproducibility
seed = 3
np.random.seed(seed)


demand = np.array([
   [0   , 100, 100 , 500 , 200 , 300, 500 , 800 , 500 , 1300, 500 , 200 , 500 , 300 , 500 , 500 , 400 , 100 , 300 , 300 , 100 , 400 , 300 , 100 ],
   [100 , 0  , 100 , 200 , 100 , 400, 200 , 400 , 200 , 600 , 200 , 100 , 300 , 100 , 100 , 400 , 200 , 0   , 100 , 100 , 0   , 100 , 0   , 0   ],
   [100 , 100, 0   , 200 , 100 , 300, 100 , 200 , 100 , 300 , 300 , 200 , 100 , 100 , 100 , 200 , 100 , 0   , 0   , 0   , 0   , 100 , 100 , 0   ],
   [500 , 200, 200 , 0   , 500 , 400, 400 , 700 , 700 , 1200, 1400, 600 , 600 , 500 , 500 , 800 , 500 , 100 , 200 , 300 , 200 , 400 , 500 , 200 ],
   [200 , 100, 100 , 500 , 0   , 200, 200 , 500 , 800 , 1000, 500 , 200 , 200 , 100 , 200 , 500 , 200 , 0   , 100 , 100 , 100 , 200 , 100 , 0   ],
   [300 , 400, 300 , 400 , 200 , 0  , 400 , 800 , 400 , 800 , 400 , 200 , 200 , 100 , 200 , 900 , 500 , 100 , 200 , 300 , 100 , 200 , 100 , 100 ],
   [500 , 200, 100 , 400 , 200 , 400, 0   , 1000, 600 , 1900, 500 , 700 , 400 , 200 , 500 , 1400, 1000, 200 , 400 , 500 , 200 , 500 , 200 , 100 ],
   [800 , 400, 200 , 700 , 500 , 800, 1000, 0   , 800 , 1600, 800 , 600 , 600 , 400 , 600 , 2200, 1400, 300 , 700 , 900 , 400 , 500 , 300 , 200 ],
   [500 , 200, 700 , 700 , 800 , 400, 600 , 800 , 0   , 2800, 1400, 600 , 600 , 600 , 900 , 1400, 900 , 200 , 400 , 600 , 300 , 700 , 500 , 200 ],
   [1300, 600, 1200, 1200, 1000, 800, 1900, 1600, 2800, 0   , 4000, 2000, 1900, 2100, 4000, 4400, 3900, 700 , 1800, 2500, 1200, 2600, 1800, 800 ],
   [500 , 200, 300 , 1500, 500 , 400, 500 , 800 , 1400, 3900, 0   , 1400, 1000, 1600, 1400, 1400, 1000, 100 , 400 , 600 , 400 , 1100, 1300, 600 ],
   [200 , 100, 200 , 600 , 200 , 200, 700 , 600 , 600 , 2000, 1400, 0   , 1300, 700 , 700 , 700 , 600 , 200 , 300 , 400 , 300 , 700 , 700 , 500 ],
   [500 , 300, 100 , 600 , 200 , 200, 400 , 600 , 600 , 1900, 1000, 1300, 0   , 600 , 700 , 600 , 500 , 100 , 300 , 600 , 600 , 1300, 800 , 800 ],
   [300 , 100, 100 , 500 , 100 , 100, 200 , 400 , 600 , 2100, 1600, 700 , 600 , 0   , 1300, 700 , 700 , 100 , 300 , 500 , 400 , 1200, 1100, 400 ],
   [500 , 100, 100 , 500 , 200 , 200, 500 , 600 , 1000, 4000, 1400, 700 , 700 , 1300, 0   , 1200, 1500, 200 , 800 , 1100, 800 , 2600, 1000, 400 ],
   [500 , 400, 200 , 800 , 500 , 900, 1400, 2200, 1400, 4400, 1400, 700 , 600 , 700 , 1200, 0   , 2800, 500 , 1300, 1600, 600 , 1200, 500 , 300 ],
   [400 , 200, 100 , 500 , 200 , 500, 1000, 1400, 900 , 3900, 1000, 600 , 500 , 700 , 1500, 2800, 0   , 600 , 1700, 1700, 600 , 1700, 600 , 300 ],
   [100 , 0  , 0   , 100 , 0   , 100, 200 , 300 , 200 , 700 , 200 , 200 , 100 , 100 , 200 , 500 , 600 , 0   , 300 , 400 , 100 , 300 , 100 , 0   ],
   [300 , 100, 0   , 200 , 100 , 200, 400 , 700 , 400 , 1800, 400 , 300 , 300 , 300 , 800 , 1300, 1700, 300 , 0   , 1200, 400 , 1200, 300 , 100 ],
   [300 , 100, 0   , 300 , 100 , 300, 500 , 900 , 600 , 2500, 600 , 500 , 600 , 500 , 1100, 1600, 1700, 400 , 1200, 0   , 1200, 2400, 700 , 400 ],
   [100 , 0  , 0   , 200 , 100 , 100, 200 , 400 , 300 , 1200, 400 , 300 , 600 , 400 , 800 , 600 , 600 , 100 , 400 , 1200, 0   , 1800, 700 , 500 ],
   [400 , 100, 100 , 400 , 200 , 200, 500 , 500 , 700 , 2600, 1100, 700 , 1300, 1200, 2600, 1200, 1700, 300 , 1200, 2400, 1800, 0   , 2100, 1100],
   [300 , 0  , 100 , 500 , 100 , 100, 200 , 300 , 500 , 1800, 1300, 700 , 800 , 1100, 1000, 500 , 600 , 100 , 300 , 700 , 700 , 2100, 0   , 700 ],
   [100 , 0  , 0   , 200 , 0   , 100, 100 , 200 , 200 , 800 , 600 , 500 , 700 , 400 , 400 , 300 , 300 , 0   , 100 , 400 , 500 , 1100, 700 , 0   ]
])

Ave = 15000
income = np.ones((24, 24)) * 39293
""" income[0, 1] = 39293 - Ave
income[4, 5] = 39293 - Ave
income[5, 1] = 39293 - Ave
income[7, 5] = 39293 - Ave
income[7, 6] = 39293 - Ave
income[8, 4] = 39293 - Ave
income[9, 8] = 39293 - Ave
income[9, 14] = 39293 + Ave
income[13, 14] = 39293 + Ave
income[14, 18] = 39293 + Ave
income[15, 7] = 39293 - Ave
income[16, 18] = 39293 + Ave
income[17, 6] = 39293 - Ave
income[18, 19] = 39293 + Ave
income[19, 17] = 39293 + Ave
income[21, 20] = 39293 + Ave
income[22, 21] = 39293 + Ave
income[23, 12] = 39293 - Ave
income[23, 20] = 39293 + Ave
income[23, 22] = 39293 - Ave
 """


income[0, 1] = 39293 + Ave
income[4, 5] = 39293 + Ave
income[5, 1] = 39293 + Ave
income[7, 5] = 39293 + Ave
income[7, 6] = 39293 + Ave
income[8, 4] = 39293 + Ave
income[9, 8] = 39293 + Ave
income[9, 14] = 39293 - Ave
income[13, 14] = 39293 - Ave
income[14, 18] = 39293 - Ave
income[15, 7] = 39293 + Ave
income[16, 18] = 39293 - Ave
income[17, 6] = 39293 + Ave
income[18, 19] = 39293 - Ave
income[19, 17] = 39293 - Ave
income[21, 20] = 39293 - Ave
income[22, 21] = 39293 - Ave
income[23, 12] = 39293 + Ave
income[23, 20] = 39293 - Ave
income[23, 22] = 39293 + Ave

free_flow_travel_time = np.array([
    [0, 3.6, 2.4, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 2.4, 2. ],
    [3.6, 0, np.inf, np.inf, np.inf, 3.0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf ],
    [2.4, np.inf, 0, 2.4, np.inf , np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 2.4, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
    [np.inf, np.inf, 2.4, 0, 1.2 , np.inf, np.inf, np.inf, np.inf, np.inf, 3.6, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
    [np.inf, np.inf, np.inf , 1.2 , 0, 2.4, np.inf, np.inf, 3.0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
    [np.inf, 3.0, np.inf, np.inf, 2.4, 0, np.inf, 1.2 , np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
    [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 0, 1.8, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1.2, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
    [np.inf, np.inf, np.inf, np.inf, np.inf, 1.2, 1.8, 0, 2.0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 3.0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
    [np.inf, np.inf, np.inf, np.inf, 3.0, np.inf, np.inf, 2.0, 0, 1.8, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
    [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1.8, 0, 3.0, np.inf, np.inf, np.inf, 3.6, 3.0, 4.2, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
    [np.inf, np.inf, np.inf, 3.6, np.inf, np.inf, np.inf, np.inf, np.inf, 3.0, 0, 3.6, np.inf, 2.4, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
    [np.inf, np.inf, 2.4, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf , 3.6, 0, 1.8, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
    [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1.8, 0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 2.4],
    [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 2.4, np.inf, np.inf, 0, 3.6, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 2.4, np.inf],
    [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 3.6, np.inf, np.inf, np.inf, 3.0, 0, np.inf, np.inf, np.inf, 2.4, np.inf, np.inf, 2.4, np.inf, np.inf],
    [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 3.0, np.inf, 3.0, np.inf, np.inf, np.inf, np.inf, np.inf, 0, 1.2, 1.8, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
    [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 4.2, np.inf, np.inf, np.inf, np.inf, np.inf, 1.2, 0, np.inf, 1.2, np.inf, np.inf, np.inf, np.inf, np.inf],
    [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1.2, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1.8, np.inf, 0, np.inf, 2.4, np.inf, np.inf, np.inf, np.inf],
    [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 2.4, np.inf, 1.2, np.inf, 0, 2.4, np.inf, np.inf, np.inf, np.inf],
    [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 2.4, 2.4, 0, 3.6, 3.0, np.inf, np.inf],
    [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 3.6, 0, 1.2, np.inf, 1.8],
    [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 2.4, np.inf, np.inf, np.inf, np.inf, 3.0, 1.2, 0, 2.4, np.inf],
    [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 2.4, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 2.4, 0, 1.2],
    [np.inf , np.inf , np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 2.4, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1.8, np.inf, 1.2, 0]
])
population_distribution = np.sum(demand, axis=1)


capacity_of_link_after_hurricane = np.zeros((24, 24))
capacity_of_link_after_hurricane[0, 1] = 6.01
capacity_of_link_after_hurricane[0, 2] = 9.01
capacity_of_link_after_hurricane[1, 0] = 12.02
capacity_of_link_after_hurricane[1, 5] = 15.92
capacity_of_link_after_hurricane[2, 0] = 46.81
capacity_of_link_after_hurricane[2, 3] = 34.22
capacity_of_link_after_hurricane[2, 11] = 6.81
capacity_of_link_after_hurricane[3, 2] = 25.82
capacity_of_link_after_hurricane[3, 4] = 28.25
capacity_of_link_after_hurricane[3, 10] = 9.04
capacity_of_link_after_hurricane[4, 3] = 46.85
capacity_of_link_after_hurricane[4, 5] = 13.86
capacity_of_link_after_hurricane[4, 8] = 10.52
capacity_of_link_after_hurricane[5, 1] = 9.92
capacity_of_link_after_hurricane[5, 4] = 9.90
capacity_of_link_after_hurricane[5, 7] = 21.62
capacity_of_link_after_hurricane[6, 7] = 15.68
capacity_of_link_after_hurricane[6, 17] = 46.81
capacity_of_link_after_hurricane[7, 5] = 9.80
capacity_of_link_after_hurricane[7, 6] = 15.68
capacity_of_link_after_hurricane[7, 8] = 10.10
capacity_of_link_after_hurricane[7, 15] = 10.09
capacity_of_link_after_hurricane[8, 4] = 20.00
capacity_of_link_after_hurricane[8, 7] = 10.10
capacity_of_link_after_hurricane[8, 9] = 27.83
capacity_of_link_after_hurricane[9, 8] = 27.83
capacity_of_link_after_hurricane[9, 10] = 20.00
capacity_of_link_after_hurricane[9, 14] = 27.02
capacity_of_link_after_hurricane[9, 15] = 10.27
capacity_of_link_after_hurricane[9, 16] = 9.99
capacity_of_link_after_hurricane[10, 3] = 9.82
capacity_of_link_after_hurricane[10, 9] = 20.00
capacity_of_link_after_hurricane[10, 11] = 9.82
capacity_of_link_after_hurricane[10, 13] = 9.75
capacity_of_link_after_hurricane[11, 2] = 46.81
capacity_of_link_after_hurricane[11, 10] = 9.82
capacity_of_link_after_hurricane[11, 12] = 51.80
capacity_of_link_after_hurricane[12, 11] = 51.80
capacity_of_link_after_hurricane[12, 23] = 10.18
capacity_of_link_after_hurricane[13, 10] = 9.75
capacity_of_link_after_hurricane[13, 14] = 10.26
capacity_of_link_after_hurricane[13, 22] = 9.85
capacity_of_link_after_hurricane[14, 9] = 7.02
capacity_of_link_after_hurricane[14, 13] = 10.26
capacity_of_link_after_hurricane[14, 18] = 9.64
capacity_of_link_after_hurricane[14, 21] = 20.63
capacity_of_link_after_hurricane[15, 7] = 10.09
capacity_of_link_after_hurricane[15, 9] = 10.27
capacity_of_link_after_hurricane[15, 16] = 10.46
capacity_of_link_after_hurricane[15, 17] = 39.36
capacity_of_link_after_hurricane[16, 9] = 9.99
capacity_of_link_after_hurricane[16, 15] = 10.46
capacity_of_link_after_hurricane[16, 18] = 9.65
capacity_of_link_after_hurricane[17, 6] = 46.81
capacity_of_link_after_hurricane[17, 15] = 39.36
capacity_of_link_after_hurricane[17, 19] = 8.11
capacity_of_link_after_hurricane[18, 14] = 4.42
capacity_of_link_after_hurricane[18, 16] = 9.65
capacity_of_link_after_hurricane[18, 19] = 10.01
capacity_of_link_after_hurricane[19, 17] = 8.11
capacity_of_link_after_hurricane[19, 18] = 6.05
capacity_of_link_after_hurricane[19, 20] = 10.12
capacity_of_link_after_hurricane[19, 21] = 10.15
capacity_of_link_after_hurricane[20, 19] = 10.12
capacity_of_link_after_hurricane[20, 21] = 10.46
capacity_of_link_after_hurricane[20, 23] = 9.77
capacity_of_link_after_hurricane[21, 14] = 20.63
capacity_of_link_after_hurricane[21, 19] = 10.15
capacity_of_link_after_hurricane[21, 20] = 10.46
capacity_of_link_after_hurricane[21, 22] = 10.00
capacity_of_link_after_hurricane[22, 13] = 9.85
capacity_of_link_after_hurricane[22, 21] = 10.00
capacity_of_link_after_hurricane[22, 23] = 10.16
capacity_of_link_after_hurricane[23, 12] = 11.38
capacity_of_link_after_hurricane[23, 20] = 9.77
capacity_of_link_after_hurricane[23, 22] = 10.16


restoration_mat = np.zeros((24, 24))
restoration_mat[0, 1] = 6.01
restoration_mat[4, 3] = 46.85
restoration_mat[5, 4] = 9.90

restoration_mat[19, 17] = 8.11
restoration_mat[9, 8] = 27.83
restoration_mat[23, 20] = 9.77
restoration_mat[2, 11] = 6.81

restoration_mat[7, 6] = 15.68
restoration_mat[8, 7] = 10.10
restoration_mat[9, 15] = 10.27
restoration_mat[9, 16] = 9.99
restoration_mat[10, 3] = 9.82
restoration_mat[10, 9] = 20.00
restoration_mat[10, 11] = 9.82
restoration_mat[10, 13] = 9.75
restoration_mat[11, 2] = 46.81
restoration_mat[11, 12] = 51.80
restoration_mat[12, 23] = 10.18

restoration_mat[13, 14] = 10.26
restoration_mat[14, 9] = 7.02
restoration_mat[14, 21] = 20.63
restoration_mat[15, 7] = 10.09

restoration_mat[18, 14] = 4.42
restoration_mat[19, 18] = 6.05

restoration_mat[20, 19] = 10.12



budget = 350

# Constants
alpha = 0.15
beta = 4
theta = 1

# Function to calculate travel time based on BPR function
def travel_time(x, c):
    return 3.6 * (1 + alpha * (x / c)**beta)

# Function to calculate the total system travel time (TSTT)
def calculate_TSTT(flow, capacity):
    TSTT = 0
    for i in range(len(flow)):
        for j in range(len(flow)):
            if capacity[i][j] > 0 and flow[i][j] > 0:
                TSTT += flow[i][j] * travel_time(flow[i][j], capacity[i][j])
    return TSTT

# Function to calculate node accessibility
def calculate_accessibility(population, travel_times):
    accessibility = np.zeros(len(population))
    for r in range(len(population)):
        for s in range(len(population)):
            if travel_times[r][s] < np.inf and travel_times[r][s] > 0:  # Handling division by zero
                accessibility[r] += population[s] / travel_times[r][s]**theta
    return accessibility

"""
# Function to calculate GINI coefficient
def calculate_GINI(accessibility):
    n = len(accessibility)
    mean_accessibility = np.mean(accessibility)
    diff_sum = np.sum([np.abs(accessibility[i] - accessibility[j]) for i in range(n) for j in range(n)])
    GINI = diff_sum / (2 * n**2 * mean_accessibility)
    return GINI
"""

# Function to calculate GINI coefficient
def calculate_GINI(income, accessibility):
    n = len(income)
    combined_values = income * accessibility  # Element-wise product
    mean_combined = np.mean(combined_values)
    diff_sum = np.sum([np.abs(combined_values[i] - combined_values[j]) for i in range(n) for j in range(n)])
    GINI = diff_sum / (2 * n**2 * mean_combined)
    return GINI


# Compute initial TSTT before restoration
initial_flow = demand
initial_capacity = free_flow_travel_time
initial_TSTT = calculate_TSTT(initial_flow, initial_capacity)

# Compute TSTT after restoration (initial guess)
restored_capacity = capacity_of_link_after_hurricane + restoration_mat
#restored_capacity = capacity_of_link_after_hurricane - restoration_mat
restored_TSTT = calculate_TSTT(initial_flow, restored_capacity)

# Calculate D (recovery deficiency index)
D = 1 - initial_TSTT / restored_TSTT
#D =  (initial_TSTT - restored_TSTT) / initial_TSTT

# Calculate E (GINI coefficient based on node accessibility)
initial_accessibility = calculate_accessibility(population_distribution, free_flow_travel_time)
restored_accessibility = calculate_accessibility(population_distribution, restored_capacity)
E = calculate_GINI(income, restored_accessibility)

# Define the upper-level objective function coefficients
mu = 0.2

# Create a CQM model
cqm = dimod.ConstrainedQuadraticModel()

# Add variables for the capacity to be restored on each link
num_nodes = len(capacity_of_link_after_hurricane)
variables = {}
for i in range(num_nodes):
    for j in range(num_nodes):
        if capacity_of_link_after_hurricane[i, j] > 0:
            var_name = f"c_{i}_{j}"
            variables[(i, j)] = dimod.Real(var_name)
            # Adding bounds for each variable
            cqm.add_constraint(variables[(i, j)] <= restoration_mat[i, j])
            cqm.add_constraint(variables[(i, j)] >= 0)

# Add the objective function
objective = sum(mu * D * variables[var] + (1 - mu) * E * variables[var] for var in variables)
cqm.set_objective(objective)

# Add the budget constraint
budget_constraint = sum(variables[var] for var in variables)
cqm.add_constraint(budget_constraint <= budget, label='budget_constraint')

# Solve the CQM using D-Wave's hybrid solver
#sampler = LeapHybridCQMSampler()
sampler = LeapHybridCQMSampler(seed=seed)
sampleset = sampler.sample_cqm(cqm)

# Extract the best solution
best_solution = sampleset.first.sample

# Compute optimal R, D, E using the best restoration plan
optimal_restored_capacity = np.zeros_like(capacity_of_link_after_hurricane)
total_cost = 0
for var, value in best_solution.items():
    if value > 0:
        i, j = map(int, var.split('_')[1:])
        if total_cost + value <= budget:
            optimal_restored_capacity[i][j] = value
            total_cost += value

# Function to minimize TSTT using User Equilibrium (UE)
def user_equilibrium(flow, capacity):
    def obj_func(f):
        return calculate_TSTT(f.reshape(flow.shape), capacity)
    
    constraints = []
    for r in range(len(demand)):
        for s in range(len(demand)):
            if r != s and demand[r][s] > 0:
                constraints.append({
                    'type': 'eq',
                    'fun': lambda f, r=r, s=s: np.sum(f.reshape(flow.shape)[r]) - demand[r][s]
                })
    
    bounds = [(0, np.inf) for _ in range(flow.size)]
    result = minimize(obj_func, flow.flatten(), bounds=bounds, constraints=constraints)
    return result.x.reshape(flow.shape)

# Compute the optimal traffic flow using User Equilibrium (UE) with the optimal restored capacity
optimal_flow = user_equilibrium(initial_flow, optimal_restored_capacity)
optimal_restored_TSTT = calculate_TSTT(optimal_flow, optimal_restored_capacity)
#optimal_D = 1 - initial_TSTT / optimal_restored_TSTT
optimal_D = (initial_TSTT - optimal_restored_TSTT) / initial_TSTT
optimal_restored_accessibility = calculate_accessibility(population_distribution, optimal_restored_capacity)
optimal_E = calculate_GINI(income, optimal_restored_accessibility)
optimal_R = mu * optimal_D + (1 - mu) * optimal_E


""" #This shows the recovery capacities for all the links.
#Output the optimized recovered capacity for each link
print("Optimized recovered capacity for each link:")
for i in range(num_nodes):
    for j in range(num_nodes):
        if capacity_of_link_after_hurricane[i, j] > 0:
            print(f"Link ({i+1}, {j+1}): {optimal_restored_capacity[i][j]:.2f}") 

 """
 # Output the solution
print("Optimized recovered capacity for each link:")
links = [(0, 1), (2, 11), (4, 3), (5, 4), (7, 6), (8, 7), (9, 8), (9, 15), (9, 16), (10, 3), (10, 9), (10, 11), (10, 13), (11, 2), (11, 12), (12, 23), (13, 14), (14, 9), (14, 21), (15,7), (18, 14), (19, 18), (19, 17), (20, 19), (23, 20)]
for (i, j) in links:
    print(f"Link ({i+1}, {j+1}): {optimal_restored_capacity[i][j]}")
  


# Output the optimal values for R, D, E, and total cost
print("\nOptimal values:")
print(f"R: {optimal_R:.4f}")
print(f"D: {optimal_D:.4f}")
print(f"E: {optimal_E:}")
print(f"Total cost: {total_cost:.2f}")

"""
# Print solve time details
print("\nSolve time details:")
print(f"QPU access time: {sampleset.info.get('qpu_access_time', 'N/A')} microseconds")
print(f"Charge time: {sampleset.info.get('charge_time', 'N/A')} microseconds")
print(f"Run time: {sampleset.info.get('run_time', 'N/A')} microseconds")
#print(f"Total solve time: {sampleset.info.get('total_real_time', 'N/A')} microseconds")

"""


