import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_theme(style='dark')

def load_tlc_data(borough='Manhattan', freq='1D'):
    taxi_zone_lookup = pd.read_csv('tlc/taxi+_zone_lookup.csv', sep=',')
    taxi_zone_lookup = taxi_zone_lookup.dropna()
    taxi_zone_lookup['LocationID'] = taxi_zone_lookup['LocationID'].astype(int)
    taxi_zone_lookup.set_index('LocationID', inplace=True)
    fhv_trips = pd.read_csv('tlc/fhv_tripdata_2021-01.csv', sep=',')
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
    b = np.zeros(shape=(T, n), dtype=np.float64)

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


