import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cvxpy as cp
import seaborn as sns
import data
import argparse
import itertools
import collections

from scipy import sparse

MEDIUM_SIZE = 24
SMALL_SIZE = 0.8 * MEDIUM_SIZE
SMALLER_SIZE = 0.6 * SMALL_SIZE
BIGGER_SIZE = 1.5 * MEDIUM_SIZE

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALLER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALLER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALLER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='tlc', choices=['tlc', 'synthetic', 'synthetic_lp', 'venmo', 'safegraph'], help='Dataset to run experiments on (see available choices)')
    parser.add_argument('--filename', type=str, default='data/venmo_jul_2018.json', help='Filename of Venmo dataset (only for venmo data)')
    parser.add_argument('--num_iters', type=int, default=1, help='Number of runs of the algorithm')
    parser.add_argument('-B', type=str, default='0', help='Total budget')
    parser.add_argument('-L', type=int, default=0, help='Bailout value (if -1 sets it to B). Safegraph data have custom bailouts')
    parser.add_argument('--method', type=str, default='fractional', choices=['fractional', 'discrete'], help='Bailout method')
    parser.add_argument('--gini', default=1, type=float, help='Gini coefficient upper bound')
    parser.add_argument('--verbose', action='store_true', help='Verbose flag for solver')
    parser.add_argument('--gini_type', type=str, default='sgc', choices=['sgc', 'standard'], help='Type of gini coefficient constraint')
    parser.add_argument('--solver', type=str, default='ECOS', help='Solver to use from cvxpy solvers')
    parser.add_argument('--no_surplus_budget', action='store_true', help='No Surplus budget flag')
    return parser.parse_args()

def get_mean_std(x):
    x = np.array(x)
    return x.mean(0), x.std(0)

def generic_gini_coefficient(n, W, row_sum, col_sum, z_bar):
    varpi_bar = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            varpi_bar[i, j] = np.abs(z_bar[i, 0] - z_bar[j, 0])
    num = np.sum(W * varpi_bar)
    den = np.sum((row_sum + col_sum) * z_bar[:, 0])

    if np.isclose(den, 0):
        return 0
    else:
        return num / den

def generate_generic_gini_coefficient_constraints(n, W, row_sum, col_sum, z_bar, varpi_bar, gini):
    return cp.sum(cp.multiply(W, varpi_bar)) <= gini * cp.sum(cp.multiply(row_sum + col_sum, z_bar[:, 0]))

def single_period_clearing(L_inst, b_inst, c_inst, B, n, L_bailouts, verbose=False, gini=1, gini_type='sgc', solver='ECOS'):
    p = (b_inst + L_inst.sum(-1)).reshape((n, 1))
    p_bar = cp.Variable((n, 1))
    z_bar = cp.Variable((n, 1))
    c_inst = c_inst.reshape((n, 1))
    objective = cp.Maximize(cp.sum(p_bar))
    A_inst = np.copy(L_inst)

    if gini < np.inf:
        varpi_bar = cp.Variable((n, n))

    for i in range(n):
        A_inst[i, :] /= p[i, 0]

    # A_inst = sparse.csr_matrix(A_inst)

    constraints = [p_bar >= 0, z_bar >= 0, z_bar <= L_bailouts, cp.sum(z_bar) <= B, p_bar <= p, p_bar <= A_inst.T @ p_bar + c_inst + z_bar]

    beta_inst = A_inst.sum(-1)
    col_sum = A_inst.sum(0)

    if gini < 1:
        constraints.append(varpi_bar >= 0)
        for i in range(n):
            for j in range(n):
                constraints.append(-varpi_bar[i, j] <= z_bar[i, 0] - z_bar[j, 0])
                constraints.append(z_bar[i, 0] - z_bar[j, 0] <= varpi_bar[i, j])
        if gini_type == 'sgc':
            constraints.append(generate_generic_gini_coefficient_constraints(n, A_inst, beta_inst, col_sum, z_bar, varpi_bar, gini))
        elif gini_type == 'standard':
            constraints.append(generate_generic_gini_coefficient_constraints(n, 1.0 - np.eye(n), (n - 1) * np.ones(n), (n - 1) * np.ones(n), z_bar, varpi_bar, gini))

    prob = cp.Problem(objective, constraints)
    result = prob.solve(verbose=verbose, solver=solver)
    beta_max = beta_inst.max()

    if gini_type == 'sgc':
        gc = generic_gini_coefficient(n, A_inst, beta_inst, col_sum, z_bar.value)
    elif gini_type == 'standard':
        gc = generic_gini_coefficient(n, 1 - np.eye(n), (n - 1) * np.ones(n), (n - 1) * np.ones(n), z_bar.value)

    zz = np.zeros_like(z_bar.value)
    zz[0] = 1
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d[i, j] = abs(zz[i, 0] - zz[j, 0])

    num = np.sum(A_inst * d)
    den = np.sum((A_inst.sum(0) + A_inst.sum(-1)) * zz[:, 0])

    if verbose:

        print('Primal program')
        print('Clearing payments', p_bar.value.T)
        print('Bailouts', z_bar.value.T)

        print('Dual program')
        print('Solvent nodes', np.isclose(p_bar.value, p))
        print('Bailout constraint', constraints[2].dual_value)
        print('Solvency constraint', constraints[3].dual_value)
        print('Default constraint', constraints[4].dual_value)

    beta_inst = beta_inst.reshape(n)

    surplus_assets = c_inst + z_bar.value + A_inst.T @ p_bar.value - p_bar.value

    return p_bar.value, z_bar.value, p, result, beta_inst, beta_max, gc, surplus_assets[:, 0]

def sequential_clearing(L, b, xi, B, n, T, L_bailouts, method='fractional', verbose=False, gini=1, gini_type='sgc', solver='ECOS', surplus_budget=False):
    if method == 'fractional':
        p_bar = np.zeros((T, n, 1))
        z_bar = np.zeros((T, n, 1))
        L_bar = np.zeros((T, n, n))
        p = np.zeros((T, n, 1))
        beta = np.zeros((T, n))
        c = np.zeros((T+1, n))
        beta_max = np.zeros((T, 1))
        rewards = np.zeros(T)
        gcs = np.zeros(T)

        budget = B * np.ones(T+1)

        for t in range(T):
            if t == 0:
                c[t, :] = xi[t, :]
                p_bar[t, :, :], z_bar[t, :, :], p[t, :, :], rewards[t], beta[t, :], beta_max[t], gcs[t], c[t+1,:] = single_period_clearing(L[t, :, :], b[t, :], c[t, :], budget[t], n, L_bailouts[t, :, :], verbose, gini, gini_type, solver)
            else:
                # Calculate uncleared liabilities
                L_bar[t, :, :] = L_bar[t - 1, :, :] + L[t, :, :]

                c[t, :] += xi[t, :]

                p_bar[t, :, :], z_bar[t, :, :], p[t, :, :], rewards[t], beta[t, :], beta_max[t], gcs[t], c[t+1,:] = single_period_clearing(L_bar[t, :, :], b[t, :], xi[t, :], budget[t], n, L_bailouts[t, :, :], verbose, gini, gini_type, solver)

            if surplus_budget:
                budget[t+1] += budget[t] - z_bar[t, :, 0].sum()

            for i in range(n):
                for j in range(n):
                    L_bar[t, i, j] = L[t, i, j] * (1 - p_bar[t, i, 0] / p[t, i, 0])
        cum_reward = np.cumsum(rewards)

        return p_bar, z_bar, cum_reward, beta, beta_max, gcs, c
    elif method == 'discrete':
        # Solve fractional problem
        p_bar, z_bar, cum_reward, beta, beta_max, gcs, c = sequential_clearing(L, b, xi, B, n, T, L_bailouts, method='fractional', verbose=verbose, gini=gini, gini_type=gini_type, solver=solver, surplus_budget=surplus_budget)

        z_bar_rounded = np.zeros_like(z_bar)

        # Rounding
        while True:
            for t in range(T):
                for i in range(n):
                    z_bar_rounded[t, i, 0] = 1.0 * np.random.binomial(L_bailouts[t, i, 0], np.maximum(0, np.minimum(1, z_bar[t, i, 0] / L_bailouts[t, i, 0])))
            if (gini >= 1 and np.all((z_bar_rounded).sum(-1).sum(-1) <= B + np.sqrt(B))) or(gini < 1 and gcs.max() <= gini and np.all((z_bar_rounded).sum(-1).sum(-1) <= B + np.sqrt(B))):
                break

        p_bar_rounded, _, cum_reward_rounded, beta_rounded, beta_max_rounded, gcs, c_disc = sequential_clearing(L, b, xi + z_bar_rounded.reshape(xi.shape), 0, n, T, L_bailouts, method='fractional', verbose=verbose, gini=1, gini_type=gini_type, solver=solver, surplus_budget=surplus_budget)

        return p_bar_rounded, z_bar_rounded, cum_reward_rounded, beta_rounded, beta_max_rounded, gcs, c_disc

def assert_lp_condition(L, b):
    n = b.shape[-1]
    T = b.shape[0]
    assert(np.all(b > 0))
    p0 = (b[0, :] + L[0, :, :].sum(-1)).reshape((n, 1))
    A0 = np.copy(L[0, :, :])

    for i in range(n):
        A0[i, :] /= p0[i, 0]

    for t in range(1, T):
        pt = (b[t, :] + L[t, :, :].sum(-1)).reshape((n, 1))
        At = np.copy(L[t, :, :])

        for i in range(n):
            At[i, :] /= pt[i, 0]
        # assert(np.allclose(At, A0))

    return A0

if __name__ == '__main__':

    args = get_args()
    B_range = [float(x) for x in args.B.split(',')]
    betas = collections.defaultdict(list)
    betas_mean = {}
    betas_std = {}
    beta_maxs = collections.defaultdict(list)
    beta_maxs_mean = {}
    beta_maxs_std = {}
    gcs = collections.defaultdict(list)
    gcs_mean = {}
    gcs_std = {}
    p_bars = collections.defaultdict(list)
    z_bars = collections.defaultdict(list)
    cum_rewardss = collections.defaultdict(list)
    p_bars_mean = {}
    z_bars_mean = {}
    p_bars_std = {}
    z_bars_std = {}
    cum_rewardss_mean = {}
    cum_rewardss_std = {}

    for B in B_range:
        for _ in range(args.num_iters):
            if args.name == 'tlc':
                L, b, xi, loc2idx, idx2zone = data.load_tlc_data()
            elif args.name == 'synthetic':
                L, b, xi, loc2idx, idx2zone = data.generate_synthetic_data()
            elif args.name == 'synthetic_lp':
                L, b, xi, loc2idx, idx2zone = data.generate_synthetic_data_lp(T=20, n=10)
            elif args.name == 'venmo':
                L, b, xi, loc2idx, idx2zone = data.load_venmo_data(args.filename)
            elif args.name == 'safegraph':
                L, b, xi, L_bailouts, loc2idx, idx2zone = data.load_safegraph_data()

            n, T = L.shape[1], L.shape[0]

            if args.name != 'safegraph':
                if args.L > 0:
                    L_bailouts = args.L * np.ones((T, n, 1))
                else:
                    L_bailouts = B * np.ones((T, n, 1))

            p_bar, z_bar, cum_rewards, beta, beta_max, gc, _ = sequential_clearing(L, b, xi, B, n, T, L_bailouts, method=args.method, verbose=args.verbose, gini=args.gini, gini_type=args.gini_type, solver=args.solver, surplus_budget=not args.no_surplus_budget)
            p_bars[B].append(p_bar)
            z_bars[B].append(z_bar)
            cum_rewardss[B].append(cum_rewards)
            beta_maxs[B].append(beta_max)
            gcs[B].append(gc)
            betas[B].append(beta)

        p_bars_mean[B], p_bars_std[B] = get_mean_std(p_bars[B])
        z_bars_mean[B], z_bars_std[B] = get_mean_std(z_bars[B])
        cum_rewardss_mean[B], cum_rewardss_std[B] = get_mean_std(cum_rewardss[B])
        beta_maxs_mean[B], beta_maxs_std[B] = get_mean_std(beta_maxs[B])
        betas_mean[B], betas_std[B] = get_mean_std(betas[B])
        gcs_mean[B], gcs_std[B] = get_mean_std(gcs[B])

        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax2 = ax1.twinx()

        t_range = (1 + np.arange(T)).astype(str)


        ax1.set_ylabel('Clearing Payments')
        ax2.set_ylabel('Cummulative Reward', color='gold')
        ax1.set_xlabel('Time')
        plt.title('Sequential Clearing ($w(t) = {}$)'.format(B))

        idx = np.argsort(-p_bars_mean[B].sum(0).reshape(n))[:5]
        labels = [idx2zone[i] for i in idx]

        for i, label in zip(idx, labels):
            ax1.plot(t_range, p_bars_mean[B][:, i, 0], marker='o', linewidth=1, label=label)
            ax1.fill_between(t_range, p_bars_mean[B][:, i, 0] - p_bars_std[B][:, i, 0], p_bars_mean[B][:, i, 0] + p_bars_std[B][:, i, 0], alpha=0.3)

        ax2.plot(t_range, cum_rewardss_mean[B], color='gold', marker='o', linewidth=4)
        ax2.fill_between(t_range, cum_rewardss_mean[B] - cum_rewardss_std[B], cum_rewardss_mean[B] + cum_rewardss_std[B], alpha=0.3, color='gold')
        ax2.tick_params(axis='y', colors='gold')
        fig.tight_layout()
        ax1.legend().set_zorder(-np.inf)
        ax2.legend().set_zorder(-np.inf)

        plt.savefig('figures/{}_{}_sequential_clearing_{}_{}_{}.png'.format(args.method, args.name, B, args.gini, args.gini_type))

        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.set_ylabel('Bailouts')
        ax1.set_xlabel('Time')
        plt.title('Bailouts ($w(t) = {}$)'.format(B))

        idx = np.argsort(-z_bars_mean[B].sum(0).reshape(n))[:5]
        labels = [idx2zone[i] for i in idx]

        for i, label in zip(idx, labels):
            ax1.plot(t_range, z_bars_mean[B][:, i, 0], marker='o', linewidth=1, label=label)
            ax1.fill_between(t_range, np.maximum(0, z_bars_mean[B][:, i, 0] - z_bars_std[B][:, i, 0]), z_bars_mean[B][:, i, 0] +  z_bars_std[B][:, i, 0], alpha=0.3)
        ax1.legend()
        fig.tight_layout()
        plt.savefig('figures/{}_{}_sequential_clearing_bailouts_{}_{}_{}.png'.format(args.method, args.name, B, args.gini, args.gini_type))

    plt.figure(figsize=(10, 5))
    plt.ylabel('Worst financial connectivity')
    plt.ylabel('Worst financial connectivity')
    plt.xlabel('Time')
    for B in B_range:
        plt.plot(t_range, beta_maxs_mean[B][:, 0], marker='o', label='w(t) = {}'.format(B))
        plt.fill_between(t_range, beta_maxs_mean[B][:, 0] - beta_maxs_std[B][:, 0], beta_maxs_mean[B][:, 0] + beta_maxs_std[B][:, 0], alpha=0.3)

    plt.legend()
    plt.savefig('figures/{}_{}_worst_financial_connectivity_{}_{}_{}.png'.format(args.method, args.name, B, args.gini, args.gini_type))

    plt.figure(figsize=(10, 5))
    if args.gini_type == 'sgc':
        plt.title('Spatial Gini Coefficient {}'.format('($g = {}$)'.format(args.gini) if args.gini < 1 else ''))
        plt.ylabel('SGC')
    if args.gini_type == 'standard':
        plt.title('Standard Gini Coefficient {}'.format('($g = {}$)'.format(args.gini) if args.gini < 1 else ''))
        plt.ylabel('GC')

    plt.xlabel('Time')
    for B in B_range:
        plt.plot(t_range, gcs_mean[B], marker='o', label='w(t) = {}'.format(B))
        plt.fill_between(t_range, gcs_mean[B] - gcs_std[B], gcs_mean[B] + gcs_std[B], alpha=0.3)

    plt.legend()
    plt.tight_layout()
    plt.ylim(0, 1)
    plt.savefig('figures/{}_{}_gini_coefficient_{}_{}_{}.png'.format(args.method, args.name, B, args.gini, args.gini_type))

    fig, ax = plt.subplots(figsize=(5, 5))
    # plt.title('Bailouts vs. Payments')
    plt.xlabel('Payments')
    plt.ylabel('Bailouts')

    palette = itertools.cycle(sns.color_palette())

    for B in B_range:
        color_ols = next(palette)
        color_rlm = next(palette)
        p_bars_total = p_bars_mean[B].sum(0)
        z_bars_total = z_bars_mean[B].sum(0)
        R2 = np.corrcoef(p_bars_total[:, 0], z_bars_total[:, 0])[0, 1]
        sns.regplot(x=p_bars_total, y=z_bars_total, ax=ax, color=color_ols)
        sns.regplot(x=p_bars_total, y=z_bars_total, ax=ax, color=color_rlm, robust=True, scatter_kws={'alpha' : 1, 'color' : 'k'})
        red_patch = mpatches.Patch(color=color_ols, label='OLS, w(t) = {}, R2 = {}'.format(B, round(R2, 3)))
        blue_patch = mpatches.Patch(color=color_rlm, label='Robust LM, w(t) = {}'.format(B))

    plt.legend(handles=[red_patch, blue_patch])
    plt.tight_layout()
    plt.savefig('figures/{}_{}_payments_vs_bailouts_{}_{}_{}.png'.format(args.method, args.name, B, args.gini, args.gini_type))

    fig, ax = plt.subplots(figsize=(5, 5))
    # plt.title('Bailouts vs. Mean Financial Connectivity')
    plt.xlabel('Mean Fin. Connectivity')
    plt.ylabel('Bailouts')

    palette = itertools.cycle(sns.color_palette())

    for B in B_range:
        color_ols = next(palette)
        color_rlm = next(palette)
        betas_total = betas_mean[B].mean(0)
        z_bars_total = z_bars_mean[B].sum(0)
        R2 = np.corrcoef(betas_total, z_bars_total[:, 0])[0, 1]
        sns.regplot(x=betas_total, y=z_bars_total, ax=ax, color=color_ols)
        sns.regplot(x=betas_total, y=z_bars_total, ax=ax, color=color_rlm, robust=True, scatter_kws={'alpha' : 1, 'color' : 'k'})
        red_patch = mpatches.Patch(color=color_ols, label='OLS, w(t) = {}, R2 = {}'.format(B, round(R2, 3)))
        blue_patch = mpatches.Patch(color=color_rlm, label='Robust LM, w(t) = {}'.format(B))

    plt.legend(handles=[red_patch, blue_patch])
    plt.tight_layout()
    plt.savefig('figures/{}_{}_betas_vs_bailouts_{}_{}_{}.png'.format(args.method, args.name, B, args.gini, args.gini_type))

    if args.gini < 1:
        objective_with_fairness = {}
        for B in B_range:
            objective_with_fairness[B] = cum_rewardss_mean[B][-1]

        cum_rewardss = collections.defaultdict(list)
        cum_rewardss_mean = {}
        cum_rewardss_std = {}

        for B in B_range:
            for _ in range(args.num_iters):
                _, _, cum_rewards, _, _, gc, _ = sequential_clearing(L, b, xi, B, n, T, L_bailouts, method=args.method, verbose=args.verbose, gini=1, gini_type=args.gini_type, surplus_budget=not args.no_surplus_budget)
                cum_rewardss[B].append(cum_rewards)

            cum_rewardss_mean[B], cum_rewardss_std[B] = get_mean_std(cum_rewardss[B])

        objective_without_fairness = {}

        for B in B_range:
            objective_without_fairness[B] = cum_rewardss_mean[B][-1]

        for B in B_range:
            print('PoF (w(t) = {}) = {}'.format(B, round(objective_without_fairness[B] / objective_with_fairness[B], 3)))
