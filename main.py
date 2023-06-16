import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from matplotlib import pyplot as plt
from options import args_parser
import os
import sys
import logging
import random
import numpy as np
import copy
import torch
import torch.backends.cudnn as cudnn
from Network.network import convnet
from local import LocalUpdate
from torch.utils.data import TensorDataset, DataLoader
from utils.getdata import process_eeg,process_hgd
from braindecode.torch_ext.util import np_to_var
from sklearn import metrics
from utils.communication import communication
from utils.FedAvg import FedAvg
from utils.get_ACC import get_ACC
from utils.Avg_center import Avgcenter
import datetime
from math import exp
import torch.nn.functional as F
from utils.Confuse_matrix import kd_loss


def test(round, save_mode_path, testset, class_num, eeg_channels, input_time_length, client_len):
    print('******begin %d round test******' % (round))

    testacc = []
    checkpoint_path = save_mode_path
    checkpoint = torch.load(checkpoint_path)
    net = convnet(class_num=class_num, eeg_channels=eeg_channels, input_time_length=input_time_length)
    for i in range(client_len):
        model = copy.deepcopy(net).cuda()
        model.load_state_dict(checkpoint['subject' + str(i + 1)])

        testdata_num = len(testset[i])

        test_loader = DataLoader(dataset=testset[i], batch_size=24, shuffle=False)

        acc = get_ACC(model=model, dataLoader=test_loader, num=testdata_num)
        testacc.append(acc)

    avgacc = testacc[0]
    for i in range(1, client_len):
        avgacc += testacc[i]
    avgacc = avgacc / client_len

    return testacc, avgacc


def vaild(round, save_mode_path, vaildset, class_num, eeg_channels, input_time_length, client_len):
    print('******begin %d round vaild******' % (round))
    # 每个客户端的验证精度列表
    vaildacc = []
    checkpoint_path = save_mode_path
    checkpoint = torch.load(checkpoint_path)
    net = convnet(class_num=class_num, eeg_channels=eeg_channels, input_time_length=input_time_length)
    for i in range(client_len):
        model = copy.deepcopy(net).cuda()
        model.load_state_dict(checkpoint['subject' + str(i + 1)])

        vailddata_num = len(vaildset[i])

        vaild_loader = DataLoader(dataset=vaildset[i], batch_size=24, shuffle=False)

        acc = get_ACC(model=model, dataLoader=vaild_loader, num=vailddata_num)
        vaildacc.append(acc)

    avgacc = vaildacc[0]
    for i in range(1, client_len):
        avgacc += vaildacc[i]
    avgacc = avgacc / client_len

    return vaildacc, avgacc


if __name__ == '__main__':
    args = args_parser()
    # save pth
    if not os.path.isdir('./result'):
        os.mkdir('./result')

    if not os.path.isdir('./result/FedEEG_Plus_result'):
        os.mkdir('./result/FedEEG_Plus_result')

    save_result_pth = './result/FedEEG_Plus_result'

    if not os.path.isdir('./pth'):
        os.mkdir('./pth')

    if not os.path.isdir('./pth/pth_FedEEG_Plus'):
        os.mkdir('./pth/pth_FedEEG_Plus')

    snapshot_path = './pth/pth_FedEEG_Plus'

    if args.dataset == '2a' or args.dataset == '2b':
        # subject = [0,1,2,3,4,5,6,7,8]
        subject = [i for i in range(9)]
    elif args.dataset == '3a':
        subject = [i for i in range(3)]
    elif args.dataset == 'hgd':
        subject = [i for i in range(14)]

    total_client_num = len(subject)

    flag_create = False

    if args.dataset == '2a' or args.dataset == '3a' or args.dataset == 'hgd':
        train_data_folder = './BCI_data/train' + args.dataset + '.mat'
        test_data_folder = './BCI_data/test' + args.dataset + '.mat'

    test_sub_acc = []
    train_set = []
    vaild_set = []
    test_set = []
    vaildres = np.zeros(shape=[0], dtype=float)
    testres = np.zeros(shape=[0], dtype=float)

    each_length = []
    beta = 0.5

    if args.dataset == '2a' or args.dataset == '3a':
        train_data_folder = './BCI_data/train' + args.dataset + '.mat'
        test_data_folder = './BCI_data/test' + args.dataset + '.mat'
        for i in subject:
            traindata, trainlabel, vailddata, vaildlabel, testdata, testlabel = process_eeg(train_data_folder,
                                                                                            test_data_folder, i + 1,
                                                                                            dataseed=args.dataset,
                                                                                            normalize_data=True)
            trainset = TensorDataset(traindata, trainlabel)
            vaildset = TensorDataset(vailddata, vaildlabel)
            testset = TensorDataset(testdata, testlabel)
            train_set.append(trainset)
            vaild_set.append(vaildset)
            test_set.append(testset)
            each_length.append(len(traindata))

    elif args.dataset == 'hgd':
        for i in subject:
            Xtrain_folder = './BCI_data/train/x' + str(i + 1) + '.npy'
            ytrain_folder = './BCI_data/train/y' + str(i + 1) + '.npy'
            Xtest_folder = './BCI_data/test/tx' + str(i + 1) + '.npy'
            ytest_folder = './BCI_data/test/ty' + str(i + 1) + '.npy'
            traindata, trainlabel, vailddata, vaildlabel, testdata, testlabel = process_hgd(Xtrain_folder,
                                                                                            ytrain_folder, Xtest_folder,
                                                                                            ytest_folder,dataseed=args.dataset,
                                                                                            normalize_data=True)
            trainset = TensorDataset(traindata, trainlabel)
            vaildset = TensorDataset(vailddata, vaildlabel)
            testset = TensorDataset(testdata, testlabel)
            train_set.append(trainset)
            vaild_set.append(vaildset)
            test_set.append(testset)
            each_length.append(len(traindata))

    print('EEGdata is Ready')
    # num of classes
    class_num = len(torch.unique(trainlabel))
    # num of channel
    eeg_channel_num = traindata.shape[2]

    eeg_data_input_time_length = traindata.shape[3]

    total_data_number = sum(each_length)

    # 每个客户端占有的数据量占比
    if args.dataset == '3a':
        client_freq = [1/3,1/3,1/3]
    else:
        client_freq = [each_length[i] / total_data_number for i in range(len(each_length))]

    if not os.path.isdir('./logs'):
        os.mkdir('./logs')

    if not os.path.isdir('./logs/Sg_pFed'):
        os.mkdir('./logs/Sg_pFed')

    if args.log_file_name is None:
        args.log_file_name = 'log-%s' % (datetime.datetime.now().strftime("%m-%d-%H-%M-%S"))
    args.logdir = 'logs/Sg_pFed'
    log_path = args.log_file_name + 'Sg_pFed.txt'
    logging.basicConfig(filename=os.path.join(args.logdir, log_path), level=logging.INFO, filemode="w",
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    # Initialize the global Network
    net_glob = convnet(class_num=class_num, eeg_channels=eeg_channel_num,
                       input_time_length=eeg_data_input_time_length)
    if len(args.gpu.split(',')) > 1:
        net_glob = torch.nn.DataParallel(net_glob, device_ids=[0, 1])

    net_glob.train()
    w_glob = net_glob.state_dict()
    w_locals = []
    trainer_locals = []
    net_locals = []
    optim_locals = []
    local_center = []
    personal_center = []
    dist_scale_f = args.dist_scale
    for i in range(len(subject)):
        local_center.append(None)
        personal_center.append(None)
    global_center = None

    for i in subject:
        index = subject.index(i)
        trainer_locals.append(LocalUpdate(args, train_set[index]))
        w_locals.append(copy.deepcopy(w_glob))
        net_locals.append(copy.deepcopy(net_glob).cuda())
        optimizer = torch.optim.Adam(net_locals[index].parameters(), lr=args.base_lr,
                                     betas=(0.9, 0.999), weight_decay=5e-4)
        optim_locals.append(copy.deepcopy(optimizer.state_dict()))

    # begin com_round
    for com_round in range(args.rounds):
        print("**********Comm round %d begins**********" % com_round)
        loss_locals = []
        args.base_lr = 0.001 * (1 + 0.01 * com_round) ** (-0.75)
        # begin local train
        for i in subject:
            idx = subject.index(i)
            # learning rate decay
            trainer_locals[idx].base_lr = 0.001 * (1 + 0.01 * com_round) ** (-0.75)
            local = trainer_locals[idx]
            optimizer = optim_locals[idx]
            w, loss, op, l_center = local.train(i + 1, args, net_locals[idx], optimizer, round=com_round,
                                                center=personal_center[i], global_center=global_center,num_classes=class_num)
            w_locals[idx] = copy.deepcopy(w)
            optim_locals[idx] = copy.deepcopy(op)
            loss_locals.append(copy.deepcopy(loss))
            local_center[idx] = copy.deepcopy(l_center)

        # update global Network and local Network
        with torch.no_grad():
            global_center = Avgcenter(local_center, client_freq)
            for i in range(len(subject)):
                personal_center[i] = beta*local_center[i]+(1-beta)*global_center

        dist_list = []
        personal_ration = []
        kl_list = []


        with torch.no_grad():
            for i in subject:
                softmax_l = torch.softmax(local_center[i].cpu(),dim=0)
                softmax_g = torch.softmax(global_center.data.cpu(), dim=0)
                kl_dis = kd_loss(softmax_l,softmax_g)
                kl_list.append(kl_dis)
                p_ration = exp(-4*kl_dis)
                personal_ration.append(p_ration)
            print(dist_list)
            clt_freq_this_round = [np.exp(-kl_list[i] * args.gamma / each_length[i]) * client_freq[i] for i in range(len(subject))]
            total = sum(clt_freq_this_round)
            clt_freq_this_meta_dist = [clt_freq_this_round[i]/total for i in range(len(subject))]
            print(clt_freq_this_meta_dist)
            # print('kl')
            # print(kl_list)
            # print('personal_ration')
            # print(personal_ration)
        '''------------------------------------'''



        if com_round > 30:
            with torch.no_grad():
                net_glob, net_locals = communication(net_glob, net_locals, clt_freq_this_meta_dist,personal_ration)
        else:
            with torch.no_grad():
                net_glob, net_locals = communication(net_glob, net_locals, client_freq,0.5)
        # with torch.no_grad():
        #     net_glob, net_locals = FedEEG_communication(net_glob, net_locals, client_freq,beta)

        loss_avg = sum(loss_locals) / len(loss_locals)
        # print(loss_avg, com_round)
        logging.info('Loss Avg {}, Round {},Learning rate{}'.format(loss_avg, com_round, args.base_lr))

        if com_round % 1 == 0:
            save_mode_path = os.path.join(snapshot_path, 'com_round_' + str(com_round) + '.pth')
            # torch.save({'state_dict': net_glob.module.state_dict(), }, save_mode_path)
            if args.dataset == '2a' or args.dataset == '2b':
                torch.save({'subject1': net_locals[0].state_dict(),
                            'subject2': net_locals[1].state_dict(),
                            'subject3': net_locals[2].state_dict(),
                            'subject4': net_locals[3].state_dict(),
                            'subject5': net_locals[4].state_dict(),
                            'subject6': net_locals[5].state_dict(),
                            'subject7': net_locals[6].state_dict(),
                            'subject8': net_locals[7].state_dict(),
                            'subject9': net_locals[8].state_dict(),
                            'state_dict': net_glob.state_dict(), }, save_mode_path)
            elif args.dataset == '3a':
                torch.save({'subject1': net_locals[0].state_dict(),
                            'subject2': net_locals[1].state_dict(),
                            'subject3': net_locals[2].state_dict(),
                            'state_dict': net_glob.state_dict(), }, save_mode_path)
            else:
                torch.save({'subject1': net_locals[0].state_dict(),
                            'subject2': net_locals[1].state_dict(),
                            'subject3': net_locals[2].state_dict(),
                            'subject4': net_locals[3].state_dict(),
                            'subject5': net_locals[4].state_dict(),
                            'subject6': net_locals[5].state_dict(),
                            'subject7': net_locals[6].state_dict(),
                            'subject8': net_locals[7].state_dict(),
                            'subject9': net_locals[8].state_dict(),
                            'subject10': net_locals[0].state_dict(),
                            'subject11': net_locals[10].state_dict(),
                            'subject12': net_locals[11].state_dict(),
                            'subject13': net_locals[12].state_dict(),
                            'subject14': net_locals[13].state_dict(),
                            'state_dict': net_glob.state_dict(), }, save_mode_path)

            logging.info("TEST Global: round: {}".format(com_round))
            # 测试
            vaildacc, avgvaildacc = vaild(com_round, save_mode_path, vaild_set, class_num, eeg_channel_num,
                                          eeg_data_input_time_length, len(subject))
            if args.dataset == '2a' or args.dataset == '2b':
                logging.info(
                    'Comm round {} Vaild result\n subject1:{:6f}\n subject2:{:6f}\n subject3:{:6f}\n subject4:{:6f}\n subject5:{:6f}'
                    '\n subject6:{:6f}\n subject7:{:6f}\n subject8:{:6f}\n subject9:{:6f}\n avgacc:{:6f}'.format(
                        com_round,
                        vaildacc[0], vaildacc[1], vaildacc[2], vaildacc[3], vaildacc[4], vaildacc[5], vaildacc[6],
                        vaildacc[7],
                        vaildacc[8], avgvaildacc))


            elif args.dataset == '3a':
                logging.info(
                    'Comm round {} Vaild result \n subject1:{:6f}\n subject2:{:6f}\n subject3:{:6f}\n avgacc:{:6f}'.format(
                        com_round, vaildacc[0], vaildacc[1], vaildacc[2], avgvaildacc))

            else:
                logging.info(
                    'Comm round {} Vaild result\n subject1:{:6f}  subject2:{:6f} subject3:{:6f}\n subject4:{:6f} subject5:{:6f}'
                    ' subject6:{:6f}\n subject7:{:6f} subject8:{:6f} subject9:{:6f}\n subject10:{:6f}  subject11:{:6f}'
                    'subject12:{:6f}\n subject13:{:6f} subject14:{:6f}  avgacc:{:6f}'.format(com_round,
                                                                                             vaildacc[0], vaildacc[1],
                                                                                             vaildacc[2], vaildacc[3],
                                                                                             vaildacc[4], vaildacc[5],
                                                                                             vaildacc[6],
                                                                                             vaildacc[7], vaildacc[8],
                                                                                             vaildacc[9], vaildacc[10],
                                                                                             vaildacc[11], vaildacc[12],
                                                                                             vaildacc[13],
                                                                                             avgvaildacc))



            testacc, avgacc = test(com_round, save_mode_path, test_set, class_num, eeg_channel_num,
                                   eeg_data_input_time_length, len(subject))
            if args.dataset == '2a' or args.dataset == '2b':
                logging.info(
                    'Comm round {} test result \n subject1:{:6f}\n subject2:{:6f}\n subject3:{:6f}\n subject4:{:6f}\n subject5:{:6f}'
                    '\n subject6:{:6f}\n subject7:{:6f}\n subject8:{:6f}\n subject9:{:6f}\n avgacc:{:6f}'.format(
                        com_round,
                        testacc[0], testacc[1], testacc[2], testacc[3], testacc[4], testacc[5], testacc[6], testacc[7],
                        testacc[8], avgacc))
            elif args.dataset == '3a':
                logging.info(
                    'Comm round {} test result \n subject1:{:6f}\n subject2:{:6f}\n subject3:{:6f}\n avgacc:{:6f}'.format(
                        com_round, testacc[0], testacc[1], testacc[2], avgacc))

            else:
                logging.info(
                    'Comm round {} Test result\n subject1:{:6f}  subject2:{:6f} subject3:{:6f}\n subject4:{:6f} subject5:{:6f}'
                    ' subject6:{:6f}\n subject7:{:6f} subject8:{:6f} subject9:{:6f}\n subject10:{:6f}  subject11:{:6f}'
                    'subject12:{:6f}\n subject13:{:6f} subject14:{:6f}  avgacc:{:6f}'.format(com_round,
                                                                                             testacc[0], testacc[1],
                                                                                             testacc[2], testacc[3],
                                                                                             testacc[4], testacc[5],
                                                                                             testacc[6],
                                                                                             testacc[7], testacc[8],
                                                                                             testacc[9], testacc[10],
                                                                                             testacc[11], testacc[12],
                                                                                             testacc[13], avgacc))

            test_sub_acc.append(testacc)
            vaildres = np.append(vaildres, avgvaildacc)

            testres = np.append(testres, avgacc)

    max_vaild_acc = np.max(vaildres)
    vaildtop = np.unique(vaildres)[-5:]
    logging.info(vaildtop)
    logging.info(max_vaild_acc)
    logging.info(np.where(vaildres == max_vaild_acc))
    for i in vaildtop:
        logging.info(np.where(vaildres == i))

    max_testacc = np.max(testres)
    top = np.unique(testres)[-5:]
    logging.info(top)
    logging.info(max_testacc)
    logging.info(np.where(testres == max_testacc))
    for i in top:
        logging.info(np.where(testres == i))
    # Take the test result of the highest validation set
    logging.info('The best result is:')
    logging.info(testres[int(np.where(vaildres == max_vaild_acc)[0])])
    logging.info(test_sub_acc[int(np.where(vaildres == max_vaild_acc)[0])])
    logging.info('Please choose the com_round' + str(np.where(vaildres == max_vaild_acc)[0]) + '_pth as the best pth')

    save_result_pth = os.path.join(save_result_pth,args.log_file_name+'-SG-pFed.txt')
    np.savetxt(save_result_pth, testres)

    plt.plot(np.arange(len(testres)), testres)
    plt.title('Testacc')
    plt.ylabel('acc')
    plt.xlabel('round')
    plt.show()
