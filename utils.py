'''
add your names and IDs here:
 * @Author: Andrew Luo 
 * @ID:20448589

'''


from tqdm import tqdm 
import torch
import time 
from datetime import timedelta


PAD, CLS = '[PAD]', '[CLS]'

# label map using one hot vector 
def label2map(class_list):
    label_map = {}
    for (i, label) in enumerate(class_list, 1):
        label_map[label] = i

    # the index should be start from 0
    for strlabel in label_map:
        label_map[strlabel] = label_map[strlabel] - 1

    return label_map

def label2int(label_map, ne): 
    label = label_map[ne]
    return label


'''
See label to int function above 
'''

def load_dataset(file_path, config):
    contents = []
    label_map = label2map(config.class_list)
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line or line == '-DOCSTART- -X- -X- O':
                continue
            content, pos, chunk, ne = line.split(' ')
            '''
            Need to rasie attention here 
            ne is string 
            need to convert label to int
            using one hot vector
            '''
            label = label2int(label_map, ne)
            token = config.tokenizer.tokenize(content)
            token = [CLS] + token
            mask = []
            seq_len = len(token)
            token_ids = config.tokenizer.convert_tokens_to_ids(token)

            pad_size = config.pad_size 

            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                    token_ids = token_ids + ([0] * (pad_size - len(token)))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size] # something like iloc 
                    seq_len = pad_size 
            
            contents.append((token_ids, int(label), seq_len, mask))

    return contents
 



def bulid_dataset(config):
    '''
    The function will output 3 outputs 
    - train dataset 
    - dev dataset 
    - test dataset 

    each output is a list contains 4 lists 
    eg:
    train_dataset = [[ids], [label], [ids_len], [mask]]

    see load_dataset function above...

    '''
    train = load_dataset(config.train_path, config)
    test = load_dataset(config.test_path, config)
    dev = load_dataset(config.dev_path, config)

    return train, dev, test 


class Datasetiterator(object):
    def __init__(self, dataset, batch_size, device):
        self.batch_size = batch_size
        self.dataset = dataset 
        self.n_batches = len(dataset) // batch_size
        self.residue = False # record num of batchs is int or not 

        if len(dataset) % self.n_batches != 0:
            self.residue = True 
        
        self.index = 0 
        self.device = device


    def _to_tensor(self, datas):
        '''
        to device means make sure that we are using the GPU if we have one 
        for more detals refer to file bertLSTMner.py 

        for pytorch if we are going to use BERT 
        We need to convert data into torch.LongTensor
        if we are using Tensorflow 
        we need to convert data into tf.record
        --Andrew 
        '''
        
        x = torch.LongTensor([item[0] for item in datas]).to(self.device)
        y = torch.LongTensor([item[1] for item in datas]).to(self.device)

        seq_len = torch.LongTensor([item[2] for item in datas]).to(self.device)
        mask = torch.LongTensor([item[3] for item in datas]).to(self.device)

        return (x, seq_len, mask), y

    # the way to iter
    # see _to_tensor function above
    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.dataset[self.index * self.batch_size : len(self.dataset)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        # in case we are getting trouble
        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration

        else:
            batches = self.dataset[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches


    def __iter__(self):
        return self
    
    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else: 
            return self.n_batches


def build_iterator(dataset, config):
    iter = Datasetiterator(dataset, config.batch_size, config.device)
    return iter 


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
