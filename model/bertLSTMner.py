'''
add your names and IDs here:
 * @Author: Andrew Luo 
 * @ID:20448589

'''
import torch
import torch.nn as nn
from pytorch_pretrained import BertModel, BertTokenizer



# the path paremeters should be something like this:
# '/data/CONLL003/train.txt'
class Config(object):
    '''Config parameters'''
    def __init__(self, dataset):
        self.model_name = 'bertLSTMner'
        
        # Data set path
        # train dataset path
        self.train_path = dataset + '/train.txt'
        # test dataset path 
        self.test_path = dataset + '/test.txt'
        # dev dataset path 
        self.dev_path = dataset + '/dev.txt'

        # label
        self.class_list = [x.strip() for x in open(dataset + '/class.txt').readlines()]
        # num of labels 
        self.num_classes = len(self.class_list)

        # saved path 
        self.save_path = '/save_dir/' + self.model_name + '.ckpt'

        # auto choose cpu or gpu 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # if batch > 1000, the performance hasn't become better, end ealier 
        self.require_improvement = 1000

        # num of epoches
        self.num_epochs = 3 # only need to train 3 times 

        # batch size 
        self.batch_size = 128

        # badding size 
        # the max length of each word 
        # long cut, short add 
        self.pad_size = 12

        # learning _rate 
        self.learning_rate = 1e-5

        # bert pre-train path 
        self.bert_path = 'BERT_BASE'

        # bert tokenization 
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)

        # bert hidden size 
        self.hidden_size = 768

        # LSTM hidden number
        self.rnn_hidden = 256
        
        # rnn layers
        self.rnn_layers = 2

        # dropout
        self.dropout = 0.5


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True # little tunning bert layer 


        # layers after BERT 
        # we add two LSTM 
        # and they are bidirectional 
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers, batch_first=True, dropout=config.dropout, bidirectional=True)
        # drop out layer
        self.dropout = nn.Dropout(config.dropout)
        # softmax layer
        self.fc = nn.Linear(config.rnn_hidden*2, config.num_classes)

    # define the forward method 
    def forward(self, x):
        # x [ ids, seq_len, mask]
        context = x[0] # matching the input sentences #shape[128, 12]
        mask = x[2] # mask for padding part #shape[128,12]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers= False)
        out, _= self.lstm(encoder_out)
        out = out[:, -1, :]
        out = self.fc(out)
        return out 

