import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import json
import networkx as nx

from ast import literal_eval
from scipy import sparse

sns.set_theme(style='dark')

def load_tlc_data(borough='Manhattan', freq='1D', eps=1):
    taxi_zone_lookup = pd.read_csv('data/tlc/taxi+_zone_lookup.csv', sep=',')
    taxi_zone_lookup = taxi_zone_lookup.dropna()
    taxi_zone_lookup['LocationID'] = taxi_zone_lookup['LocationID'].astype(int)
    taxi_zone_lookup.set_index('LocationID', inplace=True)
    fhv_trips = pd.read_csv('data/tlc/fhv_tripdata_2021-01.csv', sep=',')
    fhv_trips.dropna(subset=['PULocationID', 'DOLocationID'], how='any', inplace=True)
    fhv_trips['PULocationID'] = fhv_trips['PULocationID'].astype(int)
    fhv_trips['DOLocationID'] = fhv_trips['DOLocationID'].astype(int)
    fhv_trips = fhv_trips.join(taxi_zone_lookup, on='PULocationID', rsuffix='PU').join(taxi_zone_lookup, on='DOLocationID', rsuffix='DO')
    fhv_trips = fhv_trips[fhv_trips['DOLocationID'] != fhv_trips['PULocationID']]
    fhv_trips['pickup_datetime'] = pd.to_datetime(fhv_trips['pickup_datetime'], infer_datetime_format=True)
    fhv_trips.set_index('pickup_datetime', inplace=True)

    fhv_trips_resampled = fhv_trips.resample(freq)


    T = len(fhv_trips_resampled)
    n = len(taxi_zone_lookup[taxi_zone_lookup['Borough'] == borough])

    print('T = {}, n = {}'.format(T, n))

    loc2idx = {}
    idx2zone = {}
    for i, row in enumerate(taxi_zone_lookup[taxi_zone_lookup['Borough'] == borough].iterrows()):
        loc2idx[row[0]] = i
        idx2zone[i] = row[1]['Zone']


    P = np.zeros(shape=(T, n, n), dtype=np.float64)
    c = np.zeros(shape=(T, n), dtype=np.float64)
    b = eps * np.ones(shape=(T, n), dtype=np.float64)

    for t, (timestamp, df) in enumerate(fhv_trips_resampled):
        for i, row in enumerate(df.iterrows()):
            row = row[1]
            if borough == row['Borough'] == row['BoroughDO']:
                P[t, loc2idx[row['PULocationID']], loc2idx[row['DOLocationID']]] += 1
            elif borough == row['Borough'] and borough != row['BoroughDO']:
                b[t, loc2idx[row['PULocationID']]] += 1
            elif borough == row['BoroughDO'] and borough != row['Borough']:
                c[t, loc2idx[row['DOLocationID']]] += 1

    return P, b, c, loc2idx, idx2zone

def plot_tlc_data(borough='Manhattan', freq='1D'):
    P, b, c, loc2idx, idx2zone = load_tlc_data(borough=borough, freq=freq)
    timestamp_range = 1 + np.arange(P.shape[0])

    plt.figure(figsize=(10, 10))
    plt.xlabel('Timestamp (Days)', fontsize=16)
    plt.ylabel('# Rides', fontsize=16)
    plt.title('Aggregate Rides', fontsize=16)

    plt.plot(timestamp_range, c.sum(-1), color='g', marker='o', label='Total external inbound rides')
    plt.plot(timestamp_range, b.sum(-1), color='b', marker='o', label='Total external outbound rides')
    plt.plot(timestamp_range, P.sum(-1).sum(-1), color='k', marker='o', label='Total internal inbound/outbound rides')

    plt.legend()

    plt.savefig('aggregate_tlc.png')

    plt.figure(figsize=(10, 10))
    plt.xlabel('Timestamp (Days)', fontsize=16)
    plt.ylabel('# Rides $c(t)$', fontsize=16)
    plt.title('External inbound rides', fontsize=16)

    idxs = np.argsort(-c.sum(0))[:5]
    labels = [idx2zone[i]  for i in idxs]

    for i, label in zip(idxs, labels):
        plt.plot(timestamp_range, c[:, i], marker='o', label=label)

    plt.legend()

    plt.savefig('external_inbound_tlc.png')


    plt.figure(figsize=(10, 10))
    plt.xlabel('Timestamp (Days)', fontsize=16)
    plt.ylabel('# Rides $b(t)$', fontsize=16)
    plt.title('External outbound rides', fontsize=16)

    idxs = np.argsort(-b.sum(0))[:5]
    labels = [idx2zone[i]  for i in idxs]

    for i, label in zip(idxs, labels):
        plt.plot(timestamp_range, b[:, i], marker='o', label=label)

    plt.legend()

    plt.savefig('external_outbound_tlc.png')

    plt.figure(figsize=(10, 10))
    plt.xlabel('Timestamp (Days)', fontsize=16)
    plt.ylabel('# Rides $\sum_j p_j(t)$', fontsize=16)
    plt.title('Internal outbound rides', fontsize=16)

    idxs = np.argsort(-P.sum(0).sum(0))[:5]
    labels = [idx2zone[i]  for i in idxs]

    for i, label in zip(idxs, labels):
        plt.plot(timestamp_range, P[:, i].sum(-1), marker='o', label=label)

    plt.legend()

    plt.savefig('internal_outbound_tlc.png')

def generate_synthetic_data(n=50, T=10):

    probs = 0.1 * np.ones(shape=(T, n, n))
    probs[:, :, :10] += 0.25
    probs[:, :10, :] += 0.25

    u = np.random.uniform(size=(T, n, n))

    G = (u <= probs).astype(np.float64)

    b = np.random.exponential(1, size=(T, n))

    c = np.random.exponential(1, size=(T, n))
    L = G * np.random.exponential(1, size=(T, n, n))

    loc2idx = dict([(i, i) for i in range(n)])
    idx2zone = dict([(i, i) for i in range(n)])

    return L, b, c, loc2idx, idx2zone


def generate_synthetic_data_lp(n=2, T=10, method='deterministic'):

    # probs = 0.1 * np.ones(shape=(n, n))
    # probs[:, :10] += 0.25
    # probs[:10, :] += 0.25
    #
    # u = np.random.uniform(size=(n, n))
    # G = (u <= probs).astype(np.float64)
    G = 1 - np.eye(n).astype(np.float64)
    L = np.zeros((T, n, n)).astype(np.float64)

    if method == 'deterministic':
        for t in range(T):
            L[t, :, :] = G
        b = np.ones((T, n)).astype(np.float64)
        c = (n / 2) * np.ones((T, n)).astype(np.float64)
        c = np.zeros((T, n)).astype(np.float64)
    elif method == 'random':
        c = (n / 2) * np.random.exponential(1, size=(T, n))
        b = np.random.exponential(1, (T, n))
        A0 = np.random.dirichlet(np.ones(n), n)
        for i in range(n):
            A0[i, i] = 0

        for t in range(T):
            for i in range(n):
                R = np.eye(n).astype(np.float64)
                for j in range(n):
                    R[j, :] -= A0[i, j]
                L[t, :, :] = np.linalg.solve(R, b[t, i] * A0[i, :])

    loc2idx = dict([(i, i) for i in range(n)])
    idx2zone = dict([(i, i) for i in range(n)])

    return L, b, c, loc2idx, idx2zone

def load_venmo_data(filename):

    with open(filename) as f:
        data = json.loads(f.read())

    timestamps = set([])
    timestamp_counter = 0

    node2idx = {}
    node_counter = 0

    for key1 in data:
        for key2 in data[key1]:
            try:
                temp = literal_eval(key2)
            except:
                pass
            timestamp = temp[0]
            timestamps |= {int(timestamp)}

    timestamp2idx = dict([(t, i) for (i, t) in enumerate(sorted(list(timestamps)))])

    for key1 in data:
        for key2 in data[key1]:
            try:
                temp = literal_eval(key2)
            except:
                pass
            timestamp = temp[0]
            if len(temp) >= 2:
                u = temp[1]
            if len(temp) == 3:
                v = temp[2]

            if u not in node2idx:
                node2idx[u] = node_counter
                node_counter += 1

            if len(temp) == 3 and v not in node2idx:
                node2idx[v] = node_counter
                node_counter += 1

    idx2node = dict([(val, key) for key, val in node2idx.items()])


    n = len(node2idx)
    T = len(timestamp2idx)

    L = np.zeros((T, n, n))
    b = np.zeros((T, n))
    c = np.zeros((T, n))

    for key1 in data:
        for key2, val2 in data[key1].items():
            try:
                temp = literal_eval(key2)
            except:
                continue
            timestamp = int(temp[0])
            timestamp = timestamp2idx[timestamp]

            if len(temp) >= 2:
                u = temp[1]
                u = node2idx[u]
            if len(temp) == 3:
                v = temp[2]
                v = node2idx[v]

            if key1 == 'L':
                L[timestamp, u, v] = np.random.exponential(scale=1, size=val2).sum()
            elif key1 == 'b':
                b[timestamp, u] = np.random.exponential(scale=1, size=val2).sum()
            elif key1 == 'c':
                c[timestamp, u] = np.random.exponential(scale=1, size=val2).sum()

    b = np.maximum(1, b)

    return L, b, c, node2idx, idx2node

def load_safegraph_data():
    Gs = []
    T = 5

    for i in range(1, T + 1):
        G = nx.DiGraph(nx.read_gpickle('data/safegraph/graph_{}.gpickle'.format(i)))
        Gs.append(G)

    node2idx = {}
    counter = 0
    for G in Gs:
        for u in G.nodes():
            if u not in node2idx:
                node2idx[u] = counter
                counter += 1

    idx2node = {}

    for G in Gs:
        for u, data in G.nodes(data=True):
            u_idx = node2idx[u]

            if isinstance(u, int):
                idx2node[u_idx] = 'CBG_' + str(u)
            else:
                idx2node[u_idx] = 'POI_' + data.get('brand_name', u)

    n = len(node2idx)

    L = np.zeros((T, n, n))
    b = np.zeros((T, n))
    c = np.zeros((T, n))
    L_bailouts = np.zeros((T, n, 1))

    L_cbg_total_bailout = 0
    L_poi_total_bailout = 0
    num_cbg = 0
    num_poi = 0

    for t, G in enumerate(Gs):
        for u, data in G.nodes(data=True):
            if isinstance(u, int) and data.get('L', -1) > 0:
                num_cbg += 1
                L_cbg_total_bailout += data.get('L', 0)
            else:
                num_poi += 1
                L_poi_total_bailout += data.get('L', 0)

    L_cbg_mean_bailout = L_cbg_total_bailout / max(1, num_cbg)
    L_poi_mean_bailout = L_poi_total_bailout / max(1, num_poi)

    for t, G in enumerate(Gs):
        for u, data in G.nodes(data=True):
            b[t, node2idx[u]] = data.get('liabilities', 0)
            c[t, node2idx[u]] = data.get('assets', 0)
            L_bailouts[t, node2idx[u], 0] = data.get('L', 0)

    b = np.maximum(b, 100)

    for t, G in enumerate(Gs):
        for u, v, data in G.edges(data=True):
            L[t, node2idx[u], node2idx[v]] += data.get('weight', 0)

    for i in range(n):
        L_mean = L_bailouts[:, i].sum() / (1.0 * max(1, (L_bailouts[:, i] > 0).sum()))
        if L_mean > 0:
            for t in range(T):
                if L_bailouts[t, i] == 0:
                    L_bailouts[t, i] = L_mean
        elif idx2node[i].startswith('CBG'):
            L_bailouts[:, i] = L_cbg_mean_bailout
        else:
            L_bailouts[:, i] = L_poi_mean_bailout

    return L, b, c, L_bailouts, node2idx, idx2node
