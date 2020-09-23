import pandas as pd
import numpy as np
import copy, math


def process_communities_and_crime():
    data_path = r'../Data/communities_and_crime.csv'
    df = pd.read_csv(data_path)
    task_names = np.unique(df['state'].values)
    tasks = []
    for task_name in task_names:
        task = df[df['state'] == task_name]
        del task['state']
        ys = task['y'].values
        ans = copy.deepcopy(ys)
        median = np.median(ys)
        for i in range(len(ys)):
            if ys[i] > median:
                ans[i] = 1
            else:
                ans[i] = 0
        task['y'] = ans
        del task['x23']
        task_norm = (task - task.mean()) / task.std()
        task_norm['z'] = task['z']
        task_norm['y'] = task['y']
        tasks.append(task_norm)
    return tasks


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
    task_names = np.unique(df['location'].values)
    tasks = []
    for task_name in task_names:
        task = df[df['location'] == task_name]
        del task['location']
        task_norm = (task - task.mean()) / task.std()
        task_norm['z'] = task['z']
        task_norm['y'] = task['y']
        tasks.append(task_norm)
    return tasks


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
    task_names = np.unique(df['date'].values)
    tasks = []
    for task_name in task_names:
        task = df[df['date'] == task_name]
        del task['date']
        task_norm = (task - task.mean()) / task.std()
        task_norm['z'] = task['z']
        task_norm['y'] = task['y']
        tasks.append(task_norm)
    return tasks


def process_lsac():
    data_path = r'../Data/lsac.csv'
    df = pd.read_csv(data_path)
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    task_names = np.unique(df['grp'].values)
    tasks = []
    for task_name in task_names:
        task = df[df['grp'] == task_name]
        del task['grp']
        task_norm = (task - task.mean()) / task.std()
        task_norm['z'] = task['z']
        task_norm['y'] = task['y']
        tasks.append(task_norm)
    return tasks

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
    df['bin_dob'] = pd.cut(x=df['dob'], bins=list(range(1930, 2032, 2)), labels=list(range(50)))
    del df['dob']
    task_names = np.unique(df['bin_dob'].values)
    tasks = []
    for task_name in task_names:
        task = df[df['bin_dob'] == task_name]
        del task['bin_dob']
        task_norm = (task - task.mean()) / task.std()
        task_norm = task_norm.fillna(0)
        task_norm['z'] = task['z']
        task_norm['y'] = task['y']
        tasks.append(task_norm)
    return tasks


def split(tasks, train_num, test_num, tr_path, val_path, test_path):
    train_tasks = tasks[:train_num]
    test_tasks = tasks[train_num:(train_num + test_num)]
    val_tasks = tasks[(train_num + test_num):]
    for i in range(len(train_tasks)):
        train_task = train_tasks[i]
        if train_task.isnull().values.any():
            print("train task %s contains missing values." % (i + 1))
        train_task.to_csv(tr_path + '/task' + str(i + 1) + '.csv')
    for i in range(len(test_tasks)):
        test_task = test_tasks[i]
        if test_task.isnull().values.any():
            print("test task %s contains missing values." % (i + 1))
        test_task.to_csv(test_path + '/task' + str(i + 1) + '.csv')

    for i in range(len(val_tasks)):
        val_task = val_tasks[i]
        if val_task.isnull().values.any():
            print("val task %s contains missing values." % (i + 1))
        val_task.to_csv(val_path + '/task' + str(i + 1) + '.csv')
    print("#################################################################################################")


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


def tasks_evaluation(save, num_tasks, num_neighbors):
    total_dbc = 0
    total_discrimination = 0
    total_consistency = 0
    for task_num in range(1, num_tasks + 1):
        try:
            df = pd.read_csv(save + '/task' + str(task_num) + '.csv')
        except:
            continue
        zy = df[['z', 'y']].values
        yX = df[df.columns[2:]].values
        discrimination = cal_discrimination(zy)
        consistency = cal_consistency(yX, num_neighbors)
        dbc = cal_dbc(zy)
        print("task %s: dbc=%s, discrimination=%s, consistency=%s" % (task_num, dbc, discrimination, consistency))
        total_dbc += dbc
        total_discrimination += discrimination
        total_consistency += consistency

    print("#################################################################################################")
    print("Average dbc=%s" % (total_dbc / num_tasks))
    print("Average discrimination=%s" % (total_discrimination / num_tasks))
    print("Average consistency=%s" % (total_consistency / num_tasks))


if __name__ == "__main__":
    # process communities and crime data set
    #########################################################################################################
    # tasks = process_communities_and_crime()
    # train_percentage = 0.8
    # train_num = round(len(tasks) * train_percentage)
    # test_num = len(tasks) - train_num
    #
    # train_save_path = r'../Data/communities_and_crime/train'
    # val_save_path = r'../Data/communities_and_crime/val'
    # test_save_path = r'../Data/communities_and_crime/test'
    # split(tasks, train_num, test_num, train_save_path, val_save_path, test_save_path)
    # tasks_evaluation(test_save_path, test_num, 3)
    #########################################################################################################

    # process adult data set
    #########################################################################################################
    # tasks = process_adult()
    # train_percentage = 0.8
    # train_num = round(len(tasks) * train_percentage)
    # test_num = len(tasks) - train_num
    #
    # train_save_path = r'../Data/adult/train'
    # val_save_path = r'../Data/adult/val'
    # test_save_path = r'../Data/adult/test'
    # split(tasks, train_num, test_num, train_save_path, val_save_path, test_save_path)
    # tasks_evaluation(train_save_path, 33, 3)
    # tasks_evaluation(test_save_path, 7, 3)
    #########################################################################################################

    # process bank data set
    #########################################################################################################
    # tasks = process_bank()
    # train_percentage = 0.8
    # test_percentage = 0.1
    # train_num = round(len(tasks) * train_percentage)
    # test_num = round(len(tasks) * test_percentage)

    # train_save_path = r'../Data/bankmarketing/train'
    # val_save_path = r'../Data/bankmarketing/val'
    # test_save_path = r'../Data/bankmarketing/test'
    # split(tasks, train_num, test_num, train_save_path, val_save_path, test_save_path)
    # tasks_evaluation(train_save_path, 33, 3)
    # tasks_evaluation(test_save_path, 7, 3)
    #########################################################################################################

    # process lsac data set
    #########################################################################################################
    # tasks = process_lsac()
    # train_percentage = 0.8
    # test_percentage = 0.1
    # train_num = round(len(tasks) * train_percentage)
    # test_num = round(len(tasks) * test_percentage)
    # train_save_path = r'../Data/lsac/train'
    # val_save_path = r'../Data/lsac/val'
    # test_save_path = r'../Data/lsac/test'
    # split(tasks, train_num, test_num, train_save_path, val_save_path, test_save_path)
    #########################################################################################################

    # process compas data set
    #########################################################################################################
    tasks = process_compas()
    train_percentage = 0.8
    test_percentage = 0.1
    train_num = round(len(tasks) * train_percentage)
    test_num = round(len(tasks) * test_percentage)
    train_save_path = r'../Data/compas/train'
    val_save_path = r'../Data/compas/val'
    test_save_path = r'../Data/compas/test'
    split(tasks, train_num, test_num, train_save_path, val_save_path, test_save_path)
    #########################################################################################################