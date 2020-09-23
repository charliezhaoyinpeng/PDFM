import pandas as pd
import numpy as np
import copy, math, copy
from operator import add
from os import listdir
from os.path import isfile, join


def abs_mean_difference(input_zy):
    a, b = 0, 0
    a_size, b_size = 0, 0
    for line in input_zy:
        if line[0] == 0:
            a += line[1]
            a_size += 1
        elif line[0] == 1:
            b += line[1]
            b_size += 1
    return np.abs(a / a_size - b / b_size)


def cal_auc(input_zy):
    y_a, y_b = [], []
    a_size, b_size = 0, 0
    for line in input_zy:
        if line[0] == 0:
            y_a.append(line[1])
            a_size += 1
        elif line[0] == 1:
            y_b.append(line[1])
            b_size += 1
    count_a = 0
    for a in y_a:
        for b in y_b:
            if a >= b:
                count_a += 1
    auc_a = (count_a * 1.0) / (a_size * b_size)
    if auc_a < 0.5:
        return 1 - auc_a
    else:
        return auc_a


def cal_ir(input_zy):
    y_a, y_b = [], []
    a_size, b_size = 0, 0
    for line in input_zy:
        if line[0] == 0:
            y_a.append(line[1])
            a_size += 1
        elif line[0] == 1:
            y_b.append(line[1])
            b_size += 1
    a = np.sum(y_a) * 1.0 / a_size
    b = np.sum(y_b) * 1.0 / b_size
    if np.abs(a / b) >= 1:
        return np.abs(b / a)
    else:
        return np.abs(a / b)


def data_evaluation(df, data_name):
    zy = df[['z', 'y']].values
    md = abs_mean_difference(zy)
    auc = cal_auc(zy)
    ir = cal_ir(zy)
    print("%s: md=%s, auc=%s, ir=%s" % (data_name, md, auc, ir))


def eval_chicago_crime():
    data_path = r'../Data/chicago_crime/all'
    tasks = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    ans = []
    for task in tasks:
        df = pd.read_csv(data_path + '/' + task)
        zy = df[['z', 'y']].values
        md = abs_mean_difference(zy)
        auc = cal_auc(zy)
        ir = cal_ir(zy)
        temp = [md, auc, ir]
        if ans == []:
            ans = copy.deepcopy(temp)
        else:
            ans = [sum(i) for i in zip(ans, temp)]
    ans = np.array(ans)
    print(ans)
    ans = ans / len(tasks)
    print(ans)


def process_lsac():
    data_path = r'../Data/lsac.csv'
    df = pd.read_csv(data_path)
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    del df['grp']
    df_norm = (df - df.mean()) / df.std()
    df_norm['z'] = df['z']
    df_norm['y'] = df['y']
    df_norm.dropna()
    return df_norm


def process_compas():
    data_path = r'../Data/compas.csv'
    df = pd.read_csv(data_path)
    df['x1'] = df['x1'].astype('category')
    df['x3'] = df['x3'].astype('category')
    df['x4'] = df['x4'].astype('category')
    df['x5'] = df['x5'].astype('category')
    df['x6'] = df['x6'].astype('category')
    df['x7'] = df['x7'].astype('category')
    df['x9'] = df['x9'].astype('category')
    df['x11'] = df['x11'].astype('category')
    df['x13'] = df['x13'].astype('category')
    df['x14'] = df['x14'].astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    del df['dob']
    df_norm = (df - df.mean()) / df.std()
    df_norm['z'] = df['z']
    df_norm['y'] = df['y']
    df_norm.dropna()
    return df_norm


if __name__ == "__main__":
    # lsac_df = process_lsac()
    # data_evaluation(lsac_df, "lsac")

    # compas_df = process_compas()
    # data_evaluation(compas_df, "compas")

    eval_chicago_crime()
