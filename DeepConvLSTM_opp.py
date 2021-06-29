import numpy as np
import _pickle as cp
import matplotlib.pyplot as plt

from sliding_window import sliding_window

NB_SENSOR_CHANNELS = 113
NUM_CLASSES = 18
SLIDING_WINDOW_LENGTH = 24
# FINAL_SEQUENCE_LENGTH = 8
SLIDING_WINDOW_STEP = 12
BATCH_SIZE = 100
NUM_FILTERS = 64
FILTER_SIZE = 5
NUM_UNITS_LSTM = 128


# load sensor data

def load_dataset(filename):
    f = open(filename, 'rb')
    data = cp.load(f)
    f.close()

    X_train, y_train = data[0]
    X_test, y_test = data[1]

    print(" ..from file {}".format(filename))
    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    return X_train, y_train, X_test, y_test


# Segmentation and Reshaping


def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)


import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader


# 创建子类
class subDataset(Dataset.Dataset):
    # 初始化，定义数据内容和标签
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label

    # 返回数据集大小
    def __len__(self):
        return len(self.Data)

    # 得到数据内容和标签
    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.Tensor(self.Label[index])
        return data, label


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=NUM_FILTERS, kernel_size=(5, 1))
        self.conv2 = nn.Conv2d(NUM_FILTERS, NUM_FILTERS, (5, 1))
        self.conv3 = nn.Conv2d(NUM_FILTERS, NUM_FILTERS, (5, 1))
        self.conv4 = nn.Conv2d(NUM_FILTERS, NUM_FILTERS, (5, 1))

        # self.fc1 = nn.Linear(57856, 128)
        # self.fc2 = nn.Linear(128, 128)
        self.lstm1 = nn.LSTM(input_size=(64 * 113), hidden_size=NUM_UNITS_LSTM, num_layers=1)
        self.lstm2 = nn.LSTM(input_size=NUM_UNITS_LSTM, hidden_size=NUM_UNITS_LSTM, num_layers=1)
        self.out = nn.Linear(128 * 8, NUM_CLASSES)

    #        self.fc3 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x):
        x = (F.relu(self.conv1(x)))
        x = (F.relu(self.conv2(x)))
        x = (F.relu(self.conv3(x)))
        x = (F.relu(self.conv4(x)))
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous()
        x = x.view(-1, 8, 64 * 113)

        x = F.dropout(x, p=0.5)
        # x = F.relu(self.fc1(x))

        x, (h_n, c_n) = self.lstm1(x)
        x = F.dropout(x, p=0.5)

        # x = F.relu(self.fc2(x))
        x, (h_n, c_n) = self.lstm2(x)
        x = x.view(-1, 1 * 8 * 128)
        x = F.dropout(x, p=0.5)
        x = F.softmax(self.out(x), dim=1)
        #        x = F.relu(self.fc2(x))
        #        x = self.fc3(x)
        return x


def my_loss(outputs, targets):
    output2 = outputs - torch.max(outputs, 1, True)[0]
    P = torch.exp(output2) / torch.sum(torch.exp(output2), 1, True) + 1e-10
    loss = -torch.mean(targets * torch.log(P))
    return loss


def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        torch.nn.init.orthogonal(m.weight.data)
        torch.nn.init.orthogonal(m.bias.data)
    if classname.find('lstm') != -1:
        torch.nn.init.orthogonal(m.weight.data)
        torch.nn.init.orthogonal(m.bias.data)
    if classname.find('out') != -1:
        torch.nn.init.orthogonal(m.weight.data)
        torch.nn.init.orthogonal(m.bias.data)
    if classname.find('fc') != -1:
        torch.nn.init.orthogonal(m.weight.data)
        torch.nn.init.orthogonal(m.bias.data)


if __name__ == "__main__":

    print("Loading data...")
    X_train, y_train, X_test, y_test = load_dataset('./oppChallenge_gestures.data')
    assert NB_SENSOR_CHANNELS == X_train.shape[1]
    # Sensor data is segmented using a sliding window mechanism
    X_train, y_train = opp_sliding_window(X_train, y_train, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    X_test, y_test = opp_sliding_window(X_test, y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)

    # Data is reshaped
    X_train = X_train.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))  # for input to Conv1D
    X_test = X_test.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))  # for input to Conv1D
    # y_train = convert_to_one_hot(y_train,NUM_CLASSES) # one-hot encoding
    # y_test = convert_to_one_hot(y_test,NUM_CLASSES) # one-hot encoding
    """
    net=Net()
    t = X_train[1,:,:]
    t= t.reshape(1,1,24,113)
    t = torch.from_numpy(t)
    print(t.size())
    output = net(t)
    print(output.size())

    """
    # import random
    #
    # x_0 = list(np.array(np.where(y_train == 0))[0])
    # x_1 = list(np.array(np.where(y_train == 1))[0])
    # x_2 = list(np.array(np.where(y_train == 2))[0])
    # x_3 = list(np.array(np.where(y_train == 3))[0])
    # x_4 = list(np.array(np.where(y_train == 4))[0])
    # x_5 = list(np.array(np.where(y_train == 5))[0])
    # x_6 = list(np.array(np.where(y_train == 6))[0])
    # x_7 = list(np.array(np.where(y_train == 7))[0])
    # x_8 = list(np.array(np.where(y_train == 8))[0])
    # x_9 = list(np.array(np.where(y_train == 9))[0])
    # x_10 = list(np.array(np.where(y_train == 10))[0])
    # x_11 = list(np.array(np.where(y_train == 11))[0])
    # x_12 = list(np.array(np.where(y_train == 12))[0])
    # x_13 = list(np.array(np.where(y_train == 13))[0])
    # x_14 = list(np.array(np.where(y_train == 14))[0])
    # x_15 = list(np.array(np.where(y_train == 15))[0])
    # x_16 = list(np.array(np.where(y_train == 16))[0])
    # x_17 = list(np.array(np.where(y_train == 17))[0])
    #
    # x_0 = random.sample(x_0, 2000)  # 1600:0.8151 2000:0.8275
    # x_16 = random.sample(x_16, 800)
    # Down_sample = x_0 + x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8 + x_9 + x_10 + x_11 + x_12 + x_13 + x_14 + x_15 + x_16 + x_17
    # print("Down_sampel_size:", len(Down_sample))
    # X_train = X_train[Down_sample]
    # y_train = y_train[Down_sample]

    print(" ..after sliding and reshaping, train data: inputs {0}, targets {1}".format(X_train.shape, y_train.shape))
    print(" ..after sliding and reshaping, test data : inputs {0}, targets {1}".format(X_test.shape, y_test.shape))

    y_train = y_train.reshape(len(y_train), 1)
    # y_train= y_train.reshape(12701,1)
    y_test = y_test.reshape(9894, 1)

    net = Net().cuda()

    sample_numbers = [32348, 864, 887, 806, 846, 921, 850, 666, 628, 490, 413, 457, 457, 566, 564, 904, 3246, 623]

    weights = []

    for i in sample_numbers:
        weights.append(1000.0 / i)

    weights = torch.from_numpy(np.array(weights)).cuda()
    weights = weights.float()

    net.apply(weights_init)

    # optimizer, loss function
    optimizer = torch.optim.RMSprop(net.parameters(), lr=0.001)
    # optimizer=torch.optim.Adam(net.parameters(), lr=0.001)

    loss_F = torch.nn.CrossEntropyLoss(weight=weights)
    # loss_F=my_loss()

    # create My Dateset

    train_set = subDataset(X_train, y_train)
    test_set = subDataset(X_test, y_test)

    print(train_set.__len__())
    print(test_set.__len__())

    trainloader = DataLoader.DataLoader(train_set, batch_size=200,
                                        shuffle=True, num_workers=5)

    testloader = DataLoader.DataLoader(test_set, batch_size=100,
                                       shuffle=False, num_workers=5)

    for epoch in range(150):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            labels = labels.long()

            # wrap them in Variable
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            labels = labels.squeeze(1)
            # print(epoch,i,"inputs:",inputs.data.size(),"labels:",labels.data.size())
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            # print("labels:",labels.data.size())
            # print("outputs:",outputs.data.size())
            loss = loss_F(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')

    corret, total = 0, 0
    predicts = []
    labelss = []
    for datas, labels in testloader:
        datas = datas.cuda()
        labels = labels.cuda()
        outputs = net(datas)
        _, predicted = torch.max(outputs.data, 1)

        labels = labels.long()
        total += labels.size(0)
        corret += (predicted == labels.squeeze(1)).sum()

        predicted = predicted.cpu().numpy()
        labels = labels.cpu().squeeze(1).numpy()

        # rint(type(predicted))

        predicts = predicts + list(predicted)

        labelss = labelss + list(labels)

    import sklearn.metrics as metrics
    print("\tTest fscore:\t{:.4f} ".format(metrics.f1_score(labelss, predicts, average='weighted')))
    # print('Accuracy of the network on the testset: %d %%' % (100 * corret / total))













