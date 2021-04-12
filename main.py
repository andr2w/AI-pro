'''
add your names and IDs here:
 * @Author: Andrew Luo 
 * @ID:20448589

'''

import time 
import torch 
import numpy as np
import argparse
from model import bertLSTMner
import utils
import train

parser = argparse.ArgumentParser(description='BERT NER CLASSIFICATION')
parser.add_argument('--model', type=str, default='bertLSTMner', help='choose a model')
args = parser.parse_args()




if __name__ == '__main__':
    dataset = 'data'
    model_name = args.model
    x = bertLSTMner
    config = x.Config(dataset)

    # make sure every out come same 
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(4)
    torch.backends.cudnn.deterministic = True

    # start time # utils # show time 
    start_time = time.time()
    print('Loding the dataset now :) ..........')

    train_data, dev_data, test_data = utils.bulid_dataset(config)
    # print(train_data)
    train_iter = utils.build_iterator(train_data, config)
    dev_iter = utils.build_iterator(dev_data, config)
    test_iter = utils.build_iterator(test_data, config)

    time_dif = utils.get_time_dif(start_time)
    print('The time of preparing data: ', time_dif)

    # model Train 
    model = x.Model(config).to(config.device)
    train.train(config, model, train_iter, dev_iter, test_iter)
    





