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

FONT_SIZE = 22

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='tlc', choices=['tlc', 'synthetic', 'synthetic_lp', 'venmo', 'safegraph'])
    parser.add_argument('--filename', type=str, default='data/venmo_jul_2018.json')
    parser.add_argument('--num_iters', type=int, default=1)
    parser.add_argument('-B', type=str, default='0')
    parser.add_argument('-L', type=int, default=0)
    parser.add_argument('--method', type=str, default='fractional', choices=['fractional', 'discrete'])
    parser.add_argument('--gini', default=1, type=float)
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()

def get_mean_std(x):
    x = np.array(x)
    return x.mean(0), x.std(0)

def spatial_gini_coefficient(n, A_inst, beta_inst, z_bar):
    varpi_bar = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            varpi_bar[i, j] = np.abs(z_bar[i, 0] - z_bar[j, 0])
    num = np.sum(A_inst * varpi_bar)
    den = 2 * np.sum(beta_inst * z_bar[:, 0])
    if np.isclose(den, 0):
        return 0
    else:
        return num / den

def single_period_clearing(L_inst, b_inst, c_inst, B, n, L_bailouts, verbose=False, gini=1):
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

    A_inst = sparse.csr_matrix(A_inst)

    constraints = [p_bar >= 0, z_bar >= 0, z_bar <= L_bailouts, cp.sum(z_bar) <= B, p_bar <= p, p_bar <= A_inst.T @ p_bar + c_inst + z_bar]

    beta_inst = A_inst.sum(-1)

    if gini < 1:
        constraints.append(varpi_bar >= 0)
        for i in range(n):
            for j in range(n):
                constraints.append(-varpi_bar[i, j] <= z_bar[i, 0] - z_bar[j, 0])
                constraints.append(z_bar[i, 0] - z_bar[j, 0] <= varpi_bar[i, j])

        constraints.append(cp.sum(cp.multiply(A_inst, varpi_bar)) <= 2 * gini * cp.sum(cp.multiply(beta_inst, z_bar[:, 0])))

    prob = cp.Problem(objective, constraints)
    result = prob.solve(verbose=verbose)
    beta_max = beta_inst.max()
    sgc = spatial_gini_coefficient(n, A_inst, beta_inst, z_bar.value)

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

    return p_bar.value, z_bar.value, p, result, beta_inst, beta_max, sgc

def sequential_clearing(L, b, c, B, n, T, L_bailouts, method='fractional', verbose=False, gini=1):
    if method == 'fractional':
        p_bar = np.zeros((T, n, 1))
        z_bar = np.zeros((T, n, 1))
        L_bar = np.zeros((T, n, n))
        p = np.zeros((T, n, 1))
        beta = np.zeros((T, n))
        beta_max = np.zeros((T, 1))
        rewards = np.zeros(T)
        sgcs = np.zeros(T)

        for t in range(T):
            if t == 0:
                p_bar[t, :, :], z_bar[t, :, :], p[t, :, :], rewards[t], beta[t, :], beta_max[t], sgcs[t] = single_period_clearing(L[t, :, :], b[t, :], c[t, :], B, n, L_bailouts[t, :, :], verbose, gini)
            else:
                # Calculate uncleared liabilities
                L_bar[t, :, :] = L_bar[t - 1, :, :] + L[t, :, :]
                p_bar[t, :, :], z_bar[t, :, :], p[t, :, :], rewards[t], beta[t, :], beta_max[t], sgcs[t] = single_period_clearing(L_bar[t, :, :], b[t, :], c[t, :], B, n, L_bailouts[t, :, :], verbose, gini)

            for i in range(n):
                for j in range(n):
                    L_bar[t, i, j] = L[t, i, j] * (1 - p_bar[t, i, 0] / p[t, i, 0])
        cum_reward = np.cumsum(rewards)

        return p_bar, z_bar, cum_reward, beta, beta_max, sgcs
    elif method == 'discrete':
        # Solve fractional problem
        p_bar, z_bar, cum_reward, beta, beta_max, sgcs = sequential_clearing(L, b, c, B, n, T, L_bailouts, method='fractional', verbose=verbose, gini=gini)

        z_bar_rounded = np.zeros_like(z_bar)

        # Rounding
        while True:
            for t in range(T):
                for i in range(n):
                    z_bar_rounded[t, i, 0] = 1.0 * np.random.binomial(L_bailouts[t, i, 0], np.maximum(0, np.minimum(1, z_bar[t, i, 0] / L_bailouts[t, i, 0])))
            if (gini >= 1 and np.all((z_bar_rounded).sum(-1).sum(-1) <= B + np.sqrt(B))) or(gini < 1 and sgcs.max() <= gini and np.all((z_bar_rounded).sum(-1).sum(-1) <= B + np.sqrt(B))):
                break

        p_bar_rounded, _, cum_reward_rounded, beta_rounded, beta_max_rounded, sgcs = sequential_clearing(L, b, c + z_bar_rounded.reshape(c.shape), 0, n, T, L_bailouts, method='fractional', verbose=verbose, gini=1)

        return p_bar_rounded, z_bar_rounded, cum_reward_rounded, beta_rounded, beta_max_rounded, sgcs

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
    sgcs = collections.defaultdict(list)
    sgcs_mean = {}
    sgcs_std = {}
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
                L, b, c, loc2idx, idx2zone = data.load_tlc_data()
            elif args.name == 'synthetic':
                L, b, c, loc2idx, idx2zone = data.generate_synthetic_data()
            elif args.name == 'synthetic_lp':
                L, b, c, loc2idx, idx2zone = data.generate_synthetic_data_lp(T=20, n=10)
            elif args.name == 'venmo':
                L, b, c, loc2idx, idx2zone = data.load_venmo_data(args.filename)
            elif args.name == 'safegraph':
                L, b, c, L_bailouts, loc2idx, idx2zone = data.load_safegraph_data()

            n, T = L.shape[1], L.shape[0]

            if args.name != 'safegraph':
                if args.L > 0:
                    L_bailouts = args.L * np.ones((T, n, 1))
                else:
                    L_bailouts = B * np.ones((T, n, 1))

            p_bar, z_bar, cum_rewards, beta, beta_max, sgc = sequential_clearing(L, b, c, B, n, T, L_bailouts, method=args.method, verbose=args.verbose, gini=args.gini)
            p_bars[B].append(p_bar)
            z_bars[B].append(z_bar)
            cum_rewardss[B].append(cum_rewards)
            beta_maxs[B].append(beta_max)
            sgcs[B].append(sgc)
            betas[B].append(beta)

        p_bars_mean[B], p_bars_std[B] = get_mean_std(p_bars[B])
        z_bars_mean[B], z_bars_std[B] = get_mean_std(z_bars[B])
        cum_rewardss_mean[B], cum_rewardss_std[B] = get_mean_std(cum_rewardss[B])
        beta_maxs_mean[B], beta_maxs_std[B] = get_mean_std(beta_maxs[B])
        betas_mean[B], betas_std[B] = get_mean_std(betas[B])
        sgcs_mean[B], sgcs_std[B] = get_mean_std(sgcs[B])

        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax2 = ax1.twinx()

        t_range = (1 + np.arange(T)).astype(str)

        ax1.set_ylabel('Clearing Payments', fontsize=FONT_SIZE)
        ax2.set_ylabel('Cummulative Reward', color='gold', fontsize=FONT_SIZE)
        ax1.set_xlabel('Time', fontsize=FONT_SIZE)
        plt.title('Sequential Clearing ($B = {}$)'.format(B), fontsize=FONT_SIZE)

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

        plt.savefig('{}_{}_sequential_clearing_{}_{}.png'.format(args.method, args.name, B, args.gini))

        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.set_ylabel('Bailouts', fontsize=FONT_SIZE)
        ax1.set_xlabel('Time', fontsize=FONT_SIZE)
        plt.title('Bailouts ($B = {}$)'.format(B), fontsize=FONT_SIZE)

        idx = np.argsort(-z_bars_mean[B].sum(0).reshape(n))[:5]
        labels = [idx2zone[i] for i in idx]

        for i, label in zip(idx, labels):
            ax1.plot(t_range, z_bars_mean[B][:, i, 0], marker='o', linewidth=1, label=label)
            ax1.fill_between(t_range, np.maximum(0, z_bars_mean[B][:, i, 0] - z_bars_std[B][:, i, 0]), z_bars_mean[B][:, i, 0] +  z_bars_std[B][:, i, 0], alpha=0.3)
        ax1.legend()
        fig.tight_layout()
        plt.savefig('{}_{}_sequential_clearing_bailouts_{}_{}.png'.format(args.method, args.name, B, args.gini))

    plt.figure(figsize=(10, 5))
    plt.ylabel('Worst financial connectivity', fontsize=FONT_SIZE)
    plt.ylabel('Worst financial connectivity', fontsize=FONT_SIZE)
    plt.xlabel('Time', fontsize=FONT_SIZE)
    for B in B_range:
        plt.plot(t_range, beta_maxs_mean[B][:, 0], marker='o', label='B = {}'.format(B))
        plt.fill_between(t_range, beta_maxs_mean[B][:, 0] - beta_maxs_std[B][:, 0], beta_maxs_mean[B][:, 0] + beta_maxs_std[B][:, 0], alpha=0.3)

    plt.legend(fontsize=14)
    plt.savefig('{}_{}_worst_financial_connectivity_{}_{}.png'.format(args.method, args.name, B, args.gini))

    plt.figure(figsize=(10, 5))
    plt.title('Spatial Gini Coefficient {}'.format('($g = {}$)'.format(args.gini) if args.gini < 1 else ''), fontsize=FONT_SIZE)
    plt.ylabel('SGC', fontsize=FONT_SIZE)

    plt.xlabel('Time', fontsize=FONT_SIZE)
    for B in B_range:
        plt.plot(t_range, sgcs_mean[B], marker='o', label='B = {}'.format(B))
        plt.fill_between(t_range, sgcs_mean[B] - sgcs_std[B], sgcs_mean[B] + sgcs_std[B], alpha=0.3)

    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.ylim(0, 1)
    plt.savefig('{}_{}_spatial_gini_coefficient_{}_{}.png'.format(args.method, args.name, B, args.gini))

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.title('Bailouts vs. Payments', fontsize=FONT_SIZE)
    plt.xlabel('Payments', fontsize=FONT_SIZE)
    plt.ylabel('Bailouts', fontsize=FONT_SIZE)

    palette = itertools.cycle(sns.color_palette())

    for B in B_range:
        color_ols = next(palette)
        color_rlm = next(palette)
        p_bars_total = p_bars_mean[B].sum(0)
        z_bars_total = z_bars_mean[B].sum(0)
        R2 = np.corrcoef(p_bars_total[:, 0], z_bars_total[:, 0])[0, 1]
        sns.regplot(x=p_bars_total, y=z_bars_total, ax=ax, color=color_ols)
        sns.regplot(x=p_bars_total, y=z_bars_total, ax=ax, color=color_rlm, robust=True, scatter_kws={'alpha' : 1, 'color' : 'k'})
        red_patch = mpatches.Patch(color=color_ols, label='OLS, B = {}, R2 = {}'.format(B, round(R2, 3)))
        blue_patch = mpatches.Patch(color=color_rlm, label='Robust LM, B = {}'.format(B))

    plt.legend(handles=[red_patch, blue_patch], fontsize=FONT_SIZE)
    plt.tight_layout()
    plt.savefig('{}_{}_payments_vs_bailouts_{}_{}.png'.format(args.method, args.name, B, args.gini))

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.title('Bailouts vs. Mean Financial Connectivity', fontsize=FONT_SIZE)
    plt.xlabel('Mean Financial Connectivity', fontsize=FONT_SIZE)
    plt.ylabel('Bailouts', fontsize=FONT_SIZE)

    palette = itertools.cycle(sns.color_palette())

    for B in B_range:
        color_ols = next(palette)
        color_rlm = next(palette)
        betas_total = betas_mean[B].mean(0)
        z_bars_total = z_bars_mean[B].sum(0)
        R2 = np.corrcoef(betas_total, z_bars_total[:, 0])[0, 1]
        sns.regplot(x=betas_total, y=z_bars_total, ax=ax, color=color_ols)
        sns.regplot(x=betas_total, y=z_bars_total, ax=ax, color=color_rlm, robust=True, scatter_kws={'alpha' : 1, 'color' : 'k'})
        red_patch = mpatches.Patch(color=color_ols, label='OLS, B = {}, R2 = {}'.format(B, round(R2, 3)))
        blue_patch = mpatches.Patch(color=color_rlm, label='Robust LM, B = {}'.format(B))

    plt.legend(handles=[red_patch, blue_patch], fontsize=FONT_SIZE)
    plt.tight_layout()
    plt.savefig('{}_{}_betas_vs_bailouts_{}_{}.png'.format(args.method, args.name, B, args.gini))
