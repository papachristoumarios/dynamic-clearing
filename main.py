import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

def single_period_clearing(L_inst, b_inst, c_inst, n):
    p = (b_inst + L_inst.sum(-1)).reshape((n, 1))
    p_bar = cp.Variable((n, 1))
    c_inst = c_inst.reshape((n, 1))
    objective = cp.Maximize(cp.sum(p_bar))
    A_inst = np.copy(L_inst)


    for i in range(n):
        A_inst[i, :] /= p[i, 0]
    
    constraints = [p_bar >= 0, p_bar <= p, p_bar <= A_inst.T @ p_bar + c_inst]

    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    print(p_bar.value)

    return p_bar.value, p, result

def sequential_clearing(L, b, c, n, T):
    p_bar = np.zeros((T, n, 1))
    L_bar = np.zeros((T, n, n))
    p = np.zeros((T, n, 1))
    rewards = np.zeros(T)

    for t in range(T):
        if t == 0:
            p_bar[t, :, :], p[t, :, :], rewards[t] = single_period_clearing(L[t, :, :], b[t, :], c[t, :], n)
        else:
            # Calculate uncleared liabilities 
            L_bar[t, :, :] = L_bar[t - 1, :, :] + L[t, :, :]
            p_bar[t, :, :], p[t, :, :], rewards[t] = single_period_clearing(L_bar[t, :, :], b[t, :], c[t, :], n)

        for i in range(n):
            for j in range(n):
                L_bar[t, i, j] = L[t, i, j] * p_bar[t, i, 0] / p[t, i, 0]
    cum_reward = np.cumsum(rewards)

    return p_bar, cum_reward

def global_clearing(L, b, c, n, T):
    p_bar = cp.Variable((T, n))
    objective = cp.Maximize(cp.sum(p_bar))
    p = np.zeros((T, n))

    constraints = [p_bar >= 0]

    for t in range(T):
        if t == 0: 
            p[t, :] = (b[t, :] + L[t, :, :].sum(-1)).reshape((n, 1))
            A = L[t, :, ].copy()

            for i in range(n):
                A[i, :] /= p[t, i, 0]
        else:
            pass

        constraints.append(p_bar[t, :] <= p[t, :])
        constraints.append(p_bar[t, :] <= A.T @ p_bar[t, :] + c[t, :].reshape((n, 1)))

    prob = cp.Problem(objective, constraints)

    cum_rewards = np.cumsum(np.sum(p_bar.value.sum(0)))

    return p_bar.value, cum_rewards 


# Number of periods
T = 10 

# Number of agents
n = 10

# Topology
L = 0.5 * (np.random.uniform(size=(T, n, n)) <= 0.5).astype(np.float64) * np.random.exponential(1, size=(T, n, n))

for t in range(T):
    for i in range(n):
        L[t, i, i] = 0

b = 0.5 * np.random.exponential(1, size=(T, n))
c = np.random.exponential(1, size=(T, n))

p_bar, cum_rewards = sequential_clearing(L, b, c, n, T)

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

ax1.set_xlabel('X data')

t_range = 1 + np.arange(T)

ax1.set_ylabel('Clearing Payments')
ax2.set_ylabel('Cummulative Reward', color='g')
ax1.set_xlabel('Time')
plt.title('Sequential Clearing')

for i in range(n): 
    ax1.plot(t_range, p_bar[:, i, 0], marker='o', color='k', linewidth=1)

ax2.plot(t_range, cum_rewards, color='g', marker='o', label='Cummulative Reward', linewidth=4)
plt.show()
