# -*- coding: utf-8 -*-
# 整个模型训练测试验证代码，并保存最优模型，打印测试数据
import numpy as np
import scipy.io as sio
import time
import collections
from sklearn import metrics
from Utils import modelStatsRecord, averageAccuracy
import os
from keras import backend as K
import argparse
from Utils.CapsuleNet import CapsnetBuilder


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# setting the hyper parameters
parser = argparse.ArgumentParser(description="Capsule Network on UP.")
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--nb_classes', default=9, type=int)
parser.add_argument('--lr', default=0.001, type=float,
                    help="Initial learning rate")
parser.add_argument('--lr_decay', default=0.9, type=float,
                    help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
parser.add_argument('--lam_recon', default=0.0005, type=float,
                    help="The coefficient for the loss of decoder")
parser.add_argument('-r', '--routings', default=3, type=int,
                    help="Number of iterations used in routing algorithm. should > 0")
parser.add_argument('--debug', action='store_true',
                    help="Save weights by TensorBoard")
parser.add_argument('--save_dir', default='./result/NLCapsNet-UP-27-10%')
args = parser.parse_args()
print(args)

# if not os.path.exists(args.save_dir):
#     os.makedirs(args.save_dir)

ITER = 3
CATEGORY = 9

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

# 评价指标
KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((ITER, CATEGORY))

for index_iter in range(ITER):
    print("# %d Iteration" % (index_iter + 1))
    # 1 Iteration
    # 加载数据
    test_data = sio.loadmat(args.save_dir + '/test_data' + str(index_iter + 1) + '.mat')
    x_test = test_data['x_test']
    y_test = test_data['y_test']
    gt_test = test_data['gt_test']
    gt_test = gt_test.reshape(gt_test.shape[1:])

    # save the best validated model
    # best_weights_SEN_path = 'F:/transfer code/Tensorflow  Learning/Capsule-PCA/result/Indian_best_3D_SEN_' + str(
    #     index_iter + 1) + '.hdf5'
    best_weights_path = args.save_dir + '/UP_best_weights_' + str(index_iter + 1) + '.hdf5'

    # model_SEN = model_SEN()
    model, eval_model = CapsnetBuilder.build(input_shape=x_test.shape[1:],
                                             n_class=len(np.unique(np.argmax(y_test, 1))),
                                             routings=args.routings)

    model.load_weights(best_weights_path)

    tic7 = time.clock()
    y_pred = eval_model.predict(x_test)
    toc7 = time.clock()
    pred_test = y_pred.argmax(axis=1)
    collections.Counter(pred_test)

    print('Test time:', toc7 - tic7)

    # print(len(gt_test))
    overall_acc = metrics.accuracy_score(pred_test, gt_test)
    confusion_matrix = metrics.confusion_matrix(pred_test, gt_test)
    each_acc, average_acc = averageAccuracy.AA_andEachClassAccuracy(confusion_matrix)
    kappa = metrics.cohen_kappa_score(pred_test, gt_test)
    KAPPA.append(kappa)
    OA.append(overall_acc)
    AA.append(average_acc)
    # TRAINING_TIME_3D_SEN.append(toc6 - tic6)
    TESTING_TIME.append(toc7 - tic7)
    ELEMENT_ACC[index_iter, :] = each_acc

    print("Test finished.")
    print("# %d Iteration" % (index_iter + 1))

# 自定义输出类
modelStatsRecord.outputStats_assess(KAPPA, OA, AA, ELEMENT_ACC, TESTING_TIME, CATEGORY,
                             args.save_dir + '/UP_test.txt',
                             args.save_dir + '/UP_test_element.txt')