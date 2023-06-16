import argparse



def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
    parser.add_argument('--drop_rate', type=int, default=0.2, help='dropout rate')
    parser.add_argument('--base_lr', type=float,  default=0.001, help='maximum epoch number to train')
    parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
    parser.add_argument('--seed', type=int,  default=1337, help='random seed')
    parser.add_argument('--dataseed', type=int,  default=814, help='data random seed 2a hgd =1337,2a=814')
    parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
    parser.add_argument('--local_ep', type=int,  default=10, help='local epoch')
    parser.add_argument('--rounds', type=int,  default=1000, help='comm rounds:2a hgd =1000,3a=100')
    parser.add_argument('--dataset',type=str,default='2a',help='2a or 3a or hgd')
    parser.add_argument('--lamda_d', type=float, default=0.01, help='weight of loss_cd')
    parser.add_argument('--lamda_s', type=float, default=0.1, help='weight of loss_mse')
    parser.add_argument('--alpha', type=int, default=0.002, help='weight of alpha')
    parser.add_argument('--gamma', type=int, default=10.0, help='weight of gamma')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--logdir', type=str, default='./logs/', help='The log file name')
    parser.add_argument('--Network', type=str, default='shallow', help='which eeg Network you choose:shallow,deep,eegnet')
    parser.add_argument('--meta_round', type=int, default=3, help='number of sub-consensus groups')
    parser.add_argument('--meta_client_num', type=int, default=9, help='number of clients in each sub-consensus group')
    parser.add_argument('--dist_scale', type=float or int, default=1.0,
                        help='scale factor when computing Network distance')
    parser.add_argument('--w_mul_times', type=int, default=1, help='1')
    ### tune
    parser.add_argument('--resume', type=str,  default=None, help='Network to resume')
    parser.add_argument('--start_epoch', type=int,  default=0, help='start_epoch')
    parser.add_argument('--global_step', type=int,  default=0, help='global_step')
    parser.add_argument('--base_layers', type=int, default=9, help='the number of base layers')
    parser.add_argument('--finetune',type=bool,default=False,help='finetune or not')
    args = parser.parse_args()
    return args
