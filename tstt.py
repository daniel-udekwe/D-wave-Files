import numpy as np
import dimod
from dwave.system import LeapHybridCQMSampler
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Data provided
demand = np.array([
    [1.32, 1.32, 1.32, 1.08, 1.10, 1.25, 0.99, 0.95, 0.90, 0.90, 0.59, 0.59, 0.77, 0.74],
    [1.32, 1.25, 1.30, 1.10, 1.12, 0.90, 0.95, 0.94, 1.30, 0.59, 0.59, 0.68, 0.67, 0.59],
    [1.32, 1.25, 1.32, 1.08, 1.07, 0.95, 0.90, 0.84, 0.80, 1.62, 0.64, 0.59, 0.80, 0.80],
    [1.32, 1.30, 1.32, 1.13, 0.97, 0.91, 0.88, 0.81, 0.73, 0.80, 0.81, 0.94, 0.59, 0.59],
    [1.08, 1.10, 1.08, 1.13, 1.33, 0.90, 0.99, 1.32, 1.17, 0.95, 0.90, 0.97, 0.59, 0.59],
    [1.10, 1.12, 1.07, 0.97, 1.33, 0.94, 1.32, 1.11, 0.95, 0.74, 0.61, 1.10, 1.05, 0.00],
    [1.25, 0.90, 0.95, 0.91, 0.90, 0.94, 0.87, 0.86, 0.68, 0.59, 0.62, 0.67, 1.32, 1.32],
    [0.99, 0.95, 0.90, 0.88, 0.99, 1.32, 0.87, 1.32, 1.13, 0.95, 0.87, 0.90, 1.13, 0.00],
    [0.95, 0.94, 0.84, 0.81, 1.32, 1.11, 0.86, 1.32, 1.27, 1.14, 1.32, 0.91, 0.00, 0.00],
    [0.90, 1.30, 0.80, 0.73, 1.17, 0.95, 0.68, 1.13, 1.32, 1.32, 1.11, 1.10, 0.80, 0.00],
    [0.59, 0.59, 1.62, 0.80, 0.95, 0.74, 0.59, 0.98, 1.27, 1.32, 1.32, 1.32, 0.61, 0.00],
    [0.59, 0.68, 0.64, 0.81, 0.90, 0.61, 0.62, 0.87, 1.14, 1.11, 1.32, 1.32, 1.32, 0.00],
    [0.77, 0.67, 0.59, 0.94, 0.97, 1.10, 0.67, 0.90, 1.32, 1.32, 1.10, 1.32, 1.13, 0.00],
    [0.74, 0.59, 0.80, 0.59, 1.05, 1.32, 1.13, 0.91, 0.80, 0.61, 1.32, 1.13, 1.13, 0.00],   
])


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
capacity_of_link_after_hurricane[0, 1] = 6.02
capacity_of_link_after_hurricane[0, 2] = 9.01
capacity_of_link_after_hurricane[1, 0] = 12.02
capacity_of_link_after_hurricane[1, 5] = 15.92
capacity_of_link_after_hurricane[2, 0] = 46.81
capacity_of_link_after_hurricane[2, 3] = 34.22
capacity_of_link_after_hurricane[2, 11] = 46.81
capacity_of_link_after_hurricane[3, 2] = 25.82
capacity_of_link_after_hurricane[3, 4] = 28.25
capacity_of_link_after_hurricane[3, 10] = 9.04
capacity_of_link_after_hurricane[4, 3] = 46.85
capacity_of_link_after_hurricane[4, 5] = 13.86
capacity_of_link_after_hurricane[4, 8] = 10.52
capacity_of_link_after_hurricane[5, 1] = 9.92
capacity_of_link_after_hurricane[5, 4] = 9.90
capacity_of_link_after_hurricane[5, 7] = 21.62
capacity_of_link_after_hurricane[6, 8] = 15.68
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
capacity_of_link_after_hurricane[14, 9] = 27.02
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
restoration_mat[0, 1] = 2.01
restoration_mat[0, 2] = 3.00
restoration_mat[1, 0] = 9.28
restoration_mat[1, 5] = 5.31
restoration_mat[2, 0] = 6.66
restoration_mat[2, 3] = 5.31
restoration_mat[3, 2] = 3.00
restoration_mat[3, 4] = 9.28
restoration_mat[3, 10] = 6.66
restoration_mat[4, 3] = 15.62
restoration_mat[4, 5] = 5.31
restoration_mat[4, 8] = 3.41
restoration_mat[5, 1] = 3.31
restoration_mat[5, 4] = 9.28
restoration_mat[5, 7] = 3.00
restoration_mat[6, 8] = 5.23
restoration_mat[6, 17] = 2.01
restoration_mat[7, 5] = 5.31
restoration_mat[7, 6] = 9.28
restoration_mat[7, 8] = 2.7
restoration_mat[7, 15] = 2.01
restoration_mat[8, 4] = 5.31
restoration_mat[8, 7] = 3.00
restoration_mat[8, 9] = 6.66
restoration_mat[9, 8] = 9.28
restoration_mat[9, 10] = 6.66
restoration_mat[9, 14] = 5.31
restoration_mat[9, 15] = 2.7
restoration_mat[9, 16] = 2.01
restoration_mat[10, 3] = 3.27
restoration_mat[10, 9] = 3.00
restoration_mat[10, 11] = 2.01
restoration_mat[10, 13] = 3.00
restoration_mat[11, 2] = 3.00
restoration_mat[11, 10] = 3.27
restoration_mat[11, 12] = 17.27
restoration_mat[12, 11] = 5.31
restoration_mat[12, 23] = 3.39
restoration_mat[13, 10] = 3.00
restoration_mat[13, 14] = 2.7
restoration_mat[13, 22] = 3.00
restoration_mat[14, 9] = 6.66
restoration_mat[14, 13] = 2.01
restoration_mat[14, 18] = 3.00
restoration_mat[14, 21] = 5.31
restoration_mat[15, 7] = 3.00
restoration_mat[15, 9] = 2.7
restoration_mat[15, 17] = 17.27
restoration_mat[16, 9] = 2.01
restoration_mat[16, 15] = 2.7
restoration_mat[16, 18] = 3.00
restoration_mat[17, 6] = 6.66
restoration_mat[17, 15] = 9.28
restoration_mat[17, 19] = 2.7
restoration_mat[18, 14] = 2.01
restoration_mat[18, 16] = 2.7
restoration_mat[18, 19] = 5.31
restoration_mat[19, 17] = 17.27
restoration_mat[19, 18] = 3.00
restoration_mat[19, 20] = 2.7
restoration_mat[19, 21] = 9.28
restoration_mat[20, 19] = 3.00
restoration_mat[20, 21] = 6.66
restoration_mat[20, 23] = 2.01
restoration_mat[21, 14] = 3.00
restoration_mat[21, 19] = 17.27
restoration_mat[21, 20] = 2.7
restoration_mat[21, 22] = 3.00
restoration_mat[22, 13] = 5.31
restoration_mat[22, 21] = 9.28
restoration_mat[22, 23] = 2.7
restoration_mat[23, 12] = 5.31
restoration_mat[23, 20] = 6.66
restoration_mat[23, 22] = 2.01


budget = 100

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

# Function to calculate GINI coefficient
def calculate_GINI(accessibility):
    n = len(accessibility)
    mean_accessibility = np.mean(accessibility)
    diff_sum = np.sum([np.abs(accessibility[i] - accessibility[j]) for i in range(n) for j in range(n)])
    GINI = diff_sum / (2 * n**2 * mean_accessibility)
    return GINI

# Compute initial TSTT before restoration
initial_flow = demand
initial_capacity = free_flow_travel_time
initial_TSTT = calculate_TSTT(initial_flow, initial_capacity)


# Compute TSTT after restoration (initial guess)
#restored_capacity = capacity_of_link_after_hurricane + restoration_mat
restored_capacity = restoration_mat
restored_TSTT = calculate_TSTT(initial_flow, restored_capacity)

# Calculate D (recovery deficiency index)
D = 1 - initial_TSTT / restored_TSTT
#D =  (initial_TSTT - restored_TSTT) / initial_TSTT

# Calculate E (GINI coefficient based on node accessibility)
initial_accessibility = calculate_accessibility(population_distribution, free_flow_travel_time)
restored_accessibility = calculate_accessibility(population_distribution, restored_capacity)
E = calculate_GINI(restored_accessibility)

# Define the upper-level objective function coefficients
mu = 0.5

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
sampler = LeapHybridCQMSampler()
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
optimal_E = calculate_GINI(optimal_restored_accessibility)
optimal_R = mu * optimal_D + (1 - mu) * optimal_E


# Output the optimized recovered capacity for each link
print("Optimized recovered capacity for each link:")
for i in range(num_nodes):
    for j in range(num_nodes):
        if capacity_of_link_after_hurricane[i, j] > 0:
            print(f"Link ({i+1}, {j+1}): {optimal_restored_capacity[i][j]:.2f}")

# Output the optimal values for R, D, E, and total cost
print("\nOptimal values:")
print(f"R: {optimal_R:.4f}")
print(f"D: {optimal_D:.4f}")
print(f"E: {optimal_E:}")
print(f"Total cost: {total_cost:.2f}")

# Print solve time details
print("\nSolve time details:")
print(f"QPU access time: {sampleset.info.get('qpu_access_time', 'N/A')} microseconds")
print(f"Charge time: {sampleset.info.get('charge_time', 'N/A')} microseconds")
print(f"Run time: {sampleset.info.get('run_time', 'N/A')} microseconds")
#print(f"Total solve time: {sampleset.info.get('total_real_time', 'N/A')} microseconds")



TSTT_values = [initial_TSTT, restored_TSTT, optimal_restored_TSTT]
#TSTT_values = [optimal_restored_TSTT, restored_TSTT, optimal_restored_TSTT]
labels = ['Initial TSTT', 'TSTT after disaster', 'Optimal TSTT']
plt.figure()
plt.bar(labels, TSTT_values, color=['blue', 'orange', 'green'])
plt.ylabel('Total System Travel Time (TSTT)')
plt.title('Comparison of TSTT Before and After Optimization')
plt.show()
###############################################################################################

