import scipy.io
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.functional import normalize




'''train_ration: the proportion of train data
vaild_ration: the proportion of vaild data
'''
def process_eeg(train_path, test_path, subject_id,dataseed, train_ration=0.2, vaild_ration=0.2, normalize_data=True):
    np.random.seed(dataseed)
    torch.manual_seed(dataseed)
    all_train_data = scipy.io.loadmat(train_path)
    # 导入训练数据
    trainingdata = np.transpose(all_train_data['data' + str(subject_id) + '_X'], (2, 1, 0))
    traininglabel = np.squeeze(all_train_data['data' + str(subject_id) + '_y'])
    X_train = Variable(torch.from_numpy(trainingdata).float())
    X_train = torch.unsqueeze(X_train, dim=1)
    traininglabel = Variable(torch.from_numpy(traininglabel))

    traininglabel = torch.where(traininglabel == 1, torch.full_like(traininglabel, 0), traininglabel)
    traininglabel = torch.where(traininglabel == 2, torch.full_like(traininglabel, 1), traininglabel)
    traininglabel = torch.where(traininglabel == 3, torch.full_like(traininglabel, 2), traininglabel)
    traininglabel = torch.where(traininglabel == 4, torch.full_like(traininglabel, 3), traininglabel)

    all_test_data = scipy.io.loadmat(test_path)

    testingdata = np.transpose(all_test_data['tdata' + str(subject_id) + '_X'], (2, 1, 0))
    testinglabel = np.squeeze(all_test_data['tdata' + str(subject_id) + '_y'])
    X_test = Variable(torch.from_numpy(testingdata).float())
    X_test = torch.unsqueeze(X_test, dim=1)
    testinglabel = Variable(torch.from_numpy(testinglabel))
    testinglabel = torch.where(testinglabel == 1, torch.full_like(testinglabel, 0), testinglabel)
    testinglabel = torch.where(testinglabel == 2, torch.full_like(testinglabel, 1), testinglabel)
    testinglabel = torch.where(testinglabel == 3, torch.full_like(testinglabel, 2), testinglabel)
    testinglabel = torch.where(testinglabel == 4, torch.full_like(testinglabel, 3), testinglabel)
    if normalize_data:
        X_train = normalize(X_train, p=2, dim=3)
        X_test = normalize(X_test, p=2, dim=3)

    data = torch.cat([X_train, X_test], dim=0)
    label = torch.cat([traininglabel, testinglabel], dim=0)
    shuffle_list = [index for index in range(len(data))]
    #shuffle data
    np.random.shuffle(shuffle_list)
    data = data[shuffle_list]
    label = label[shuffle_list]
    traindata = data[:int(len(data) * train_ration)]
    trainlabel = label[:int(len(data) * train_ration)]
    vailddata = data[int(len(data) * train_ration):int(len(data) * (train_ration+vaild_ration))]
    vaildlabel = label[int(len(data) * train_ration):int(len(data) * (train_ration+vaild_ration))]
    testdata = data[int(len(data) * (train_ration+vaild_ration)):]
    testlabel = label[int(len(data) * (train_ration+vaild_ration)):]

    return traindata, trainlabel, vailddata, vaildlabel, testdata, testlabel


#
def process_hgd(Xtrain_path, ytrain_path, Xtest_path, ytest_path, dataseed,train_ration=0.2, vaild_ration=0.2,normalize_data=True):
    np.random.seed(dataseed)
    torch.manual_seed(dataseed)
    # 导入训练数据
    trainingdata = np.load(Xtrain_path)
    traininglabel = np.load(ytrain_path)
    X_train = Variable(torch.from_numpy(trainingdata).float())
    X_train = torch.unsqueeze(X_train, dim=1)
    traininglabel = Variable(torch.from_numpy(traininglabel))
    print(traininglabel.shape)

    # 导入测试数据
    testingdata = np.load(Xtest_path)
    testinglabel = np.load(ytest_path)
    X_test = Variable(torch.from_numpy(testingdata).float())
    X_test = torch.unsqueeze(X_test, dim=1)
    testinglabel = Variable(torch.from_numpy(testinglabel))
    print(testinglabel.shape)

    if normalize_data:
        X_train = normalize(X_train, p=2, dim=3)
        X_test = normalize(X_test, p=2, dim=3)

    # 将test和train的数据和标签在第0个维度上拼接
    data = torch.cat([X_train, X_test], dim=0)
    label = torch.cat([traininglabel, testinglabel], dim=0)
    shuffle_list = [index for index in range(len(data))]
    # shuffle data
    np.random.shuffle(shuffle_list)
    data = data[shuffle_list]
    label = label[shuffle_list]
    traindata = data[:int(len(data) * train_ration)]
    trainlabel = label[:int(len(data) * train_ration)]
    vailddata = data[int(len(data) * train_ration):int(len(data) * (train_ration + vaild_ration))]
    vaildlabel = label[int(len(data) * train_ration):int(len(data) * (train_ration + vaild_ration))]
    testdata = data[int(len(data) * (train_ration + vaild_ration)):]
    testlabel = label[int(len(data) * (train_ration + vaild_ration)):]

    return traindata, trainlabel, vailddata, vaildlabel, testdata, testlabel
