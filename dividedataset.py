import numpy as np
import math
# 产生新数据集的过程
# indexToAssignment(train_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
# 训练集 ，151 ，151， 3
def indexToAssignment(index_, Row, Col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index_):
        # counter 是从0开始计数的，是具体的值
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


def assignmentToIndex(assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index


def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row - ex_len, pos_row + ex_len + 1), :]
    selected_patch = selected_rows[:, range(pos_col - ex_len, pos_col + ex_len + 1)]
    return selected_patch


# divide dataset into train and test datasets
def sampling(proptionVal, groundTruth):
    labels_loc = {}
    train = {}
    test = {}
    m = max(groundTruth)
    print('total class: ', m)
    # 16
    # 16类，对每一类样本要先打乱，然后再按比例分配，得到一个字典，因为上面是枚举，所以样本和标签的对应
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        # print(indices)
        # 每一类的样本数
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = int(proptionVal * len(indices))
        train[i] = indices[:-nb_val]
        test[i] = indices[-nb_val:]
    # 将所有的训练样本存到train集合中，将所有的测试样本存到test集合中
    train_indices = []
    test_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    print('total size' + str(len(test_indices) + len(train_indices)))
    print('train size' + str(len(train_indices)))
    print('test size' + str(len(test_indices)))
    return train_indices, test_indices

def sampling_(proptionVal, groundTruth):
    labels_loc = {}
    train = {}
    test = {}
    val = {}
    m = max(groundTruth)
    print('total class: ', m)
    gt = groundTruth.ravel().tolist()

    indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x !=0]
    train_indices = [j for j in indices if j <= 104895]
    test_indices_final = [j for j in indices if j > 129465]

    for i in range(m):
        indices_up = [j for j in train_indices if gt[j] == i + 1]
        # 每一类的样本数
        np.random.shuffle(indices_up)
        labels_loc[i] = indices_up
        nb_val = int(proptionVal * len(indices_up))
        train[i] = indices_up[:-nb_val]
        test[i] = indices_up[-nb_val:]
        # 将所有的训练样本存到train集合中，将所有的测试样本存到test集合中
    train_indices = []
    test_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    print(len(test_indices))
    # 5068
    print('train size:' + str(len(train_indices)))
    # 570
    # print(len(test_indices_final))
    # 6331
    return train_indices, test_indices, test_indices_final

def sampling_new(proptionVal, groundTruth, new_indices):
    labels_loc = {}
    train = {}
    test = {}
    m = max(groundTruth)
    print('total class: ', m)
    # 16
    # 16类，对每一类样本要先打乱，然后再按比例分配，得到一个字典，因为上面是枚举，所以样本和标签的对应
    for i in range(m):
        indices = list(new_indices['new_assign_' + str(i)][-1])
        # print(indices.shape)
        # print(len(indices))
        # print(indices)
        # 每一类的样本数
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = int(proptionVal * len(indices))
        train[i] = indices[:-nb_val]
        test[i] = indices[-nb_val:]
    # 将所有的训练样本存到train集合中，将所有的测试样本存到test集合中
    train_indices = []
    test_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    print('total size' + str(len(test_indices) + len(train_indices)))
    print('train size' + str(len(train_indices)))
    print('test size' + str(len(test_indices)))
    return train_indices, test_indices