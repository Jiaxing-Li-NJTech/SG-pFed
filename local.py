from torch.utils.data import DataLoader, Dataset
import torch
import torch.optim
from options import args_parser
import copy
from utils.constraints import MaxNormDefaultConstraint
args = args_parser()
import torch.nn as nn
from utils.centerloss import Center_vector,center_loss
from utils.ger_Center_matrix import get_center_matrix


# 读取对应数据集
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        # total local_dataset
        return len(self.idxs)

    def __getitem__(self, item):
        items, index, image, label = self.dataset[self.idxs[item]]
        return items, index, image, label


# 标记客户端本地训练
class LocalUpdate(object):
    def __init__(self, args, trainset):
        self.ldr_train = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        self.epoch = 0
        self.iter_num = 0
        self.base_lr = args.base_lr

    def train(self, subject_id,args, net, op_dict,round,center,global_center,num_classes):
        global total_feature, total_label
        center_vector=copy.deepcopy(center)
        g_center = copy.deepcopy(global_center)
        net.train()
        self.optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr, betas=(0.9, 0.999), weight_decay=5e-4)
        self.optimizer.load_state_dict(op_dict)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr

        loss_fn = nn.CrossEntropyLoss()
        cd_loss = center_loss()
        mesloss = nn.MSELoss()


        # train and update
        epoch_loss = []
        print('begin sub%d training'%(subject_id))
        for epoch in range(args.local_ep):
            batch_loss = []
            for step, batch in enumerate(self.ldr_train):
                inputs, labels = batch
                input_vars, label_vars = inputs.cuda(), labels.cuda().long()
                features, preds = net(input_vars)
                if step == 0:
                    with torch.no_grad():
                        total_feature = features.data
                        total_label = label_vars.data
                elif step > 0:
                    with torch.no_grad():
                        total_feature = torch.cat([total_feature, features.data], dim=0)
                        total_label = torch.cat([total_label, label_vars.data], dim=0)


                loss_classification = loss_fn(preds,label_vars)
                if round < args.k:
                    loss = loss_classification

                else:
                    loss_cd = cd_loss(features, label_vars.view(len(label_vars), 1), center_vector.cuda().long(),num_classes)
                    batch_center = get_center_matrix(features,label_vars,NUM_CLASSES=num_classes)
                    # batch_center = Center_vector(features, label_vars.view(len(label_vars), 1))
                    loss_s = mesloss(batch_center,g_center)

                    loss = loss_classification + args.lamda_d * loss_cd+args.lamda_s*loss_s


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                model_constraint = MaxNormDefaultConstraint()
                model_constraint.apply(net)

                batch_loss.append(loss.item())

                self.iter_num = self.iter_num + 1
            if (round%2 == 0 and epoch==0):
                with torch.no_grad():
                    # center_vector = Center_vector(total_feature, total_label.view(len(total_label), 1))
                    center_vector = get_center_matrix(total_feature, total_label)
            else:
                with torch.no_grad():
                    alpha = args.alpha
                    c_t = torch.sum(center_vector, dim=0) / num_classes
                    center_vector = center_vector + alpha * (
                            (center_vector - c_t) / torch.norm(center_vector - c_t, p=2, dim=1).view((num_classes, -1)))

            self.epoch = self.epoch + 1
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

            # print("sup:", epoch_loss)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), copy.deepcopy(self.optimizer.state_dict()),copy.deepcopy(center_vector.data)
