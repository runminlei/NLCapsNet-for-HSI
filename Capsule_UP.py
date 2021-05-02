# -*- coding: utf-8 -*-
# 整个模型训练测试验证代码，并保存最优模型，打印测试数据
import numpy as np
import scipy.io as sio
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
import keras.callbacks as kcallbacks
import time
from sklearn import preprocessing
from sklearn.decomposition import PCA
from Utils import zeroPadding, CapsuleNet
import os
import matplotlib.pyplot as plt
import keras
from keras import optimizers
from keras import backend as K
import argparse
from dividedataset import indexToAssignment, selectNeighboringPatch, sampling

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# setting the hyper parameters
parser = argparse.ArgumentParser(description="Capsule Network on UP.")
parser.add_argument('--epochs', default=100, type=int)
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
parser.add_argument('--input_dimension', default=103, type=int,
                    help="Number of dimensions for input datasets.")
parser.add_argument('--save_dir', default='./result/NLCapsNet-UP-27-10%')
args = parser.parse_args()
print(args)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

PATCH_LENGTH = 13  # Patch_size (13*2+1)*(13*2+1)
img_rows, img_cols = 27, 27  # 27, 27
patience = 200
n_components=3

# 10%:10%:70% data for training, validation and testing
TOTAL_SIZE = 42776
VAL_SIZE = 4281
TRAIN_SIZE = 4281
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
VALIDATION_SPLIT = 0.9  # 20% for trainnig and 80% for validation and testing
# 0.4  25670
# 0.5  21391
# 0.6  17113
# 0.7  12838
# 0.8  8558
# 0.85 6421
# 0.9  4281
# 0.95 2144
# 0.99 432

img_channels = 103

ITER = 3
CATEGORY = 9

# 写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('capsnet_loss'))
        self.accuracy['batch'].append(logs.get('capsnet_acc'))
        self.val_loss['batch'].append(logs.get('val_capsnet_loss'))
        self.val_acc['batch'].append(logs.get('val_capsnet_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('capsnet_loss'))
        self.accuracy['epoch'].append(logs.get('capsnet_acc'))
        self.val_loss['epoch'].append(logs.get('val_capsnet_loss'))
        self.val_acc['epoch'].append(logs.get('val_capsnet_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

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

# 调用设计好的模型
def model_CAPS():
    model, eval_model, manipulate_model = CapsuleNet.CapsnetBuilder.build_capsnet(input_shape=x_train.shape[1:],
                                                                                  n_class=len(np.unique(np.argmax(y_train, 1))),
                                                                                  routings=args.routings)
    model.compile(optimizer=optimizers.Adam(lr=0.001),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., 0.392],
                  metrics={'capsnet': 'accuracy'})
    return model, eval_model, manipulate_model

# 加载数据
# 修正的Indian pines数据集
mat_data = sio.loadmat('./datasets/UP/PaviaU.mat')
data_IN = mat_data['paviaU']
# 标签数据
mat_gt = sio.loadmat('./datasets/UP/PaviaU_gt.mat')
gt_IN = mat_gt['paviaU_gt']

new_gt_IN = gt_IN

# 对数据进行reshape处理之后，进行scale操作
data = data_IN.reshape(np.prod(data_IN.shape[:2]), np.prod(data_IN.shape[2:]))
gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]), )

# 标准化操作，即将所有数据沿行沿列均归一化道0-1之间
data = preprocessing.scale(data)

pca = PCA(n_components=n_components)
data = pca.fit_transform(data)

# 对数据边缘进行填充操作
data_ = data.reshape(data_IN.shape[0], data_IN.shape[1], n_components)
whole_data = data_
padded_data = zeroPadding.zeroPadding_3D(whole_data, PATCH_LENGTH)

train_data = np.zeros((TRAIN_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1, n_components))
test_data = np.zeros((TEST_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1, n_components))

seeds = [1334, 1335, 1336]

for index_iter in range(ITER):
    print("# %d Iteration" % (index_iter + 1))
    # # 1 Iteration

    # save the best validated model
    best_weights_path = args.save_dir + '/UP_best_weights_' + str(index_iter + 1) + '.hdf5'

    # 通过sampling函数拿到测试和训练样本
    np.random.seed(seeds[index_iter])
    train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)
    # train_indices 5418     test_indices 48711

    # gt本身是标签类，从标签类中取出相应的标签 -1，转成one-hot形式
    y_train = gt[train_indices] - 1
    y_train = to_categorical(np.asarray(y_train))

    y_test = gt[test_indices] - 1
    y_test = to_categorical(np.asarray(y_test))

    # 这个地方论文也解释了一下，是新建了一个以采集中心为主的新数据集，还是对元数据集进行了一些更改
    train_assign = indexToAssignment(train_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(train_assign)):
        train_data[i] = selectNeighboringPatch(padded_data, train_assign[i][0], train_assign[i][1], PATCH_LENGTH)

    test_assign = indexToAssignment(test_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(test_assign)):
        test_data[i] = selectNeighboringPatch(padded_data, test_assign[i][0], test_assign[i][1], PATCH_LENGTH)

    # 拿到了新的数据集进行reshpae之后，数据处理就结束了
    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], n_components)
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], n_components)

    # 在测试数据集上进行验证和测试的划分
    x_val = x_test_all[-VAL_SIZE:]
    y_val = y_test[-VAL_SIZE:]

    x_test = x_test_all[:-VAL_SIZE]
    y_test = y_test[:-VAL_SIZE]

    gt_test = gt[test_indices] - 1
    gt_test = gt_test[:-VAL_SIZE]
    sio.savemat(args.save_dir + '/test_data' + str(index_iter + 1), {'x_test': x_test, 'y_test': y_test, 'gt_test': gt_test})
    sio.savemat(args.save_dir + '/train_data' + str(index_iter + 1), {'x_train': x_train, 'y_train': y_train})
    sio.savemat(args.save_dir + '/val_data' + str(index_iter + 1), {'x_val': x_val, 'y_val': y_val})

    ############################################################################################################
    model, eval_model, manipulate_model = model_CAPS()

    # 创建一个实例history
    history = LossHistory()

    # callbacks
    log = kcallbacks.CSVLogger(args.save_dir + '/log' + str(index_iter + 1) + '.csv')
    # tb = kcallbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
    #                             batch_size=args.batch_size, histogram_freq=int(args.debug))
    # monitor：监视数据接口，此处是val_loss,patience是在多少步可以容忍没有提高变化
    earlyStopping6 = kcallbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='auto')
    # 用户每次epoch最后都会保存模型，如果save_best_only=True,那么最近验证误差最后的数据将会被保存下来
    saveBestModel6 = kcallbacks.ModelCheckpoint(best_weights_path, monitor='val_loss', verbose=1,
                                                save_best_only=True,
                                                mode='auto')
    lr_decay = kcallbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    print(x_train.shape)
    model.compile(optimizer=Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})
    print(x_train.shape, x_test.shape)


    # 训练和验证
    tic6 = time.clock()
    history_3d_SEN = model.fit(
        [x_train, y_train], [y_train, x_train],
        validation_data=[[x_val, y_val], [y_val, x_val]],
        batch_size=args.batch_size,
        epochs=args.epochs, shuffle=True, callbacks=[log, lr_decay, earlyStopping6, saveBestModel6, history])
    toc6 = time.clock()

    print('Training Time: ', toc6 - tic6)

    # 绘制acc-loss曲线
    history.loss_plot('epoch')

    print("Training Finished.")
    print("# %d Iteration" % (index_iter + 1))