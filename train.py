'''
add your names and IDs here:
 * @Author: Andrew Luo 
 * @ID:20448589

'''
import time 
import utils
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as f 
from sklearn import metrics
from pytorch_pretrained.optimization import BertAdam
from tensorboardX import SummaryWriter



def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    # start batch normalization and dropout 
    model.train()
    # get the parameters of the model
    param_optimizer = list(model.named_parameters())
    # parameters that no need to decay 
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any (nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(params= optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter)*config.num_epochs)

    total_batch = 0 # record how may batchs
    dev_best_loss = float('inf')
    last_imporve = 0 # record last time dev loss 
    flag = False
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime())) 
    model.train()

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}'.format(epoch+1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = f.cross_entropy(outputs, labels)
            loss.backward(retain_graph=True)
            optimizer.step()

            '''
            cuz we are using sklearn 
            so the output we need to let it into our cpu
            than we can plot/print mertics 
            '''

            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predict = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predict)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss 
                    torch.save(model.state_dict(), config.save_path)
                    imporve = '*'
                    last_imporve = total_batch
                else:
                    imporve = ' '

                time_dif = utils.get_time_dif(start_time)
                msg = 'Iter:{0:6}, Train Loss:{1:5.2}, Train Acc:{2:>6.2}, Val Loss:{3:>5.2}, Val acc: {4:>6.2%}, Time:{5} {6}'
                print(msg.format(total_batch,
                                 loss.item(),
                                 train_acc,
                                 dev_loss,
                                 dev_acc,
                                 time_dif,
                                 imporve))

                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            
            total_batch = total_batch + 1
            if total_batch - last_imporve > config.require_improvement:
                '''
                the model has not imporve for a long time 
                shut it end 
                use less
                '''
                print('No optimization for a long time, Auto ending!')
                flag = True
                break
        if flag:
            break 

    test(config, model, test_iter)

def evaluate(config, model, dev_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in dev_iter:
            outputs = model(texts)
            loss = f.cross_entropy(outputs, labels)
            loss_total = loss_total + loss 
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:

        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)

        return acc, loss_total / len(dev_iter), report, confusion

    return acc, loss_total / len(dev_iter)


def test(config, model, test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2}, Test Acc:{1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print('Precision, Recall and F1-Score')
    print(test_report)
    print('Confusion Matrix')
    print(test_confusion)
    time_dif = utils.get_time_dif(start_time)
    print('Using time:', time_dif)






