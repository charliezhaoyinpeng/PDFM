import pandas as pd
import numpy as np
import copy, math


def cal_discrimination(input_zy):
    a_values = []
    b_values = []

    for line in input_zy:
        if line[0] == 0:
            a_values.append(line[1])
        elif line[0] == 1:
            b_values.append(line[1])

    if len(a_values) == 0:
        discrimination = sum(b_values) * 1.0 / len(b_values)
    elif len(b_values) == 0:
        discrimination = sum(a_values) * 1.0 / len(a_values)
    else:
        discrimination = sum(a_values) * 1.0 / len(a_values) - sum(b_values) * 1.0 / len(b_values)
    return abs(discrimination)


# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)


# Locate the most similar neighbors
# example: get_neighbors(yX, X[0], 3)
def get_neighbors(yX, target_row, num_neighbors):
    distances = list()
    for yX_row in yX:
        X_row = yX_row[1:]
        y = yX_row[0]
        dist = euclidean_distance(target_row, X_row)
        distances.append((y, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


def cal_consistency(yX, num_neighbors):
    ans = 0
    for yX_row in yX:
        temp = 0
        target_row = yX_row[1:]
        target_y = yX_row[0]
        y_neighbors = get_neighbors(yX, target_row, num_neighbors)
        for y_neighbor in y_neighbors:
            temp += abs(target_y - y_neighbor)
        ans += temp
    return (1 - (ans * 1.0) / (len(yX) * num_neighbors))


def cal_dbc(input_zy):
    length = len(input_zy)
    z_bar = np.mean(input_zy[:, 0])
    dbc = 0
    for zy in input_zy:
        dbc += (zy[0] - z_bar) * zy[1] * 1.0
    return abs(dbc / length)


def data_evaluation(df, data_name, num_neighbors):
    zy = df[['z', 'y']].values
    yX = df[df.columns[2:]].values
    discrimination = cal_discrimination(zy)
    consistency = cal_consistency(yX, num_neighbors)
    dbc = cal_dbc(zy)
    print("%s: dbc=%s, discrimination=%s, consistency=%s" % (data_name, dbc, discrimination, consistency))


def process_communities_and_crime():
    data_path = r'../Data/communities_and_crime.csv'
    df = pd.read_csv(data_path)
    del df['state']
    del df['x23']
    ys = df['y'].values
    ans = copy.deepcopy(ys)
    median = np.median(ys)
    for i in range(len(ys)):
        if ys[i] > median:
            ans[i] = 1
        else:
            ans[i] = 0
    df['y'] = ans
    df_norm = (df - df.mean()) / df.std()
    df_norm['z'] = df['z']
    df_norm['y'] = df['y']
    return df_norm

def process_adult():
    data_path = r'../Data/adult.csv'
    df = pd.read_csv(data_path)
    df['x2'] = df['x2'].astype('category')
    df['x4'] = df['x4'].astype('category')
    df['x6'] = df['x6'].astype('category')
    df['x7'] = df['x7'].astype('category')
    df['x8'] = df['x8'].astype('category')
    df['x9'] = df['x9'].astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    del df['location']
    df_norm = (df - df.mean()) / df.std()
    df_norm['z'] = df['z']
    df_norm['y'] = df['y']
    return df_norm

def process_bank():
    data_path = r'../Data/bankmarketing.csv'
    df = pd.read_csv(data_path)
    df['x2'] = df['x2'].astype('category')
    df['x3'] = df['x3'].astype('category')
    df['x4'] = df['x4'].astype('category')
    df['x5'] = df['x5'].astype('category')
    df['x6'] = df['x6'].astype('category')
    df['x7'] = df['x7'].astype('category')
    df['x12'] = df['x12'].astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    del df['date']
    df_norm = (df - df.mean()) / df.std()
    df_norm['z'] = df['z']
    df_norm['y'] = df['y']
    return df_norm


if __name__ == "__main__":
    num_neighbors = 3

    ccdf = process_communities_and_crime()
    data_evaluation(ccdf, "communities_and_crime", num_neighbors)

    adf = process_adult()
    data_evaluation(adf, "adult", num_neighbors)

    bdf = process_bank()
    data_evaluation(bdf, "bank marketing", num_neighbors)
