import torch
from torch import nn
import numpy as np
import torch.utils.data as Data
import os
import torch.backends.cudnn as cudnn
# from thop import profile
# from thop import clever_format
import torch.optim as optim

from SENET import *
from torchstat import stat
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES']='0'

# parser = argparse.ArgumentParser(description='PyTorch Har Training')
# parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
# parser.add_argument('--resume', '-r', action='store_true',
#                     help='resume from checkpoint')
# args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


print('==> Preparing data..')

train_x = np.load('/mnt/ba3b04da-ce1b-4c21-ad1b-3aff7d337cdf/gaowenbin/Project/DeepConvLSTM/opp_dataset_1s/opp_train_x_1s.npy')
shape = train_x.shape
train_x = torch.from_numpy(np.reshape(train_x.astype(np.float), [shape[0], shape[1], shape[2], 1]))
train_x = train_x.type(torch.FloatTensor).cuda()

train_y = (np.load('/mnt/ba3b04da-ce1b-4c21-ad1b-3aff7d337cdf/gaowenbin/Project/DeepConvLSTM/opp_dataset_1s/opp_train_y_1s.npy'))
train_y = torch.from_numpy(train_y)
train_y = train_y.type(torch.FloatTensor).cuda()


test_x = np.load('/mnt/ba3b04da-ce1b-4c21-ad1b-3aff7d337cdf/gaowenbin/Project/DeepConvLSTM/opp_dataset_1s/opp_test_x_1s.npy')
test_x = torch.from_numpy(np.reshape(test_x.astype(np.float), [test_x.shape[0], test_x.shape[1], test_x.shape[2], 1]))
test_x = test_x.type(torch.FloatTensor)

test_y = np.load('/mnt/ba3b04da-ce1b-4c21-ad1b-3aff7d337cdf/gaowenbin/Project/DeepConvLSTM/opp_dataset_1s/opp_test_y_1s.npy')
test_y = torch.from_numpy(test_y.astype(np.float32))
test_y = test_y.type(torch.FloatTensor)



print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)
trainset = Data.TensorDataset(train_x, train_y)
trainloader = Data.DataLoader(dataset=trainset, batch_size=200, shuffle=True, num_workers=0)

testset = Data.TensorDataset(test_x, test_y)
testloader = Data.DataLoader(dataset=testset, batch_size=7915, shuffle=True, num_workers=0)

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # define an empty for Conv_ReLU_MaxPool
        # net = []

        # block 1
        self.conv1 = nn.Conv2d(in_channels=30, out_channels=128, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)

        self.shortcut1 = nn.Conv2d(in_channels=30, out_channels=128, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.sbn1 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(True)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(1, 1),padding=(1, 0))
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(True)

        self.shortcut2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.sbn2 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.bn5 = nn.BatchNorm2d(384)
        self.relu5 = nn.ReLU(True)
        self.conv6 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn6 = nn.BatchNorm2d(384)
        self.relu6 = nn.ReLU(True)

        self.shortcut3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.sbn3 = nn.BatchNorm2d(384)

        self.fc = nn.Linear(5760, num_classes)


    def forward(self, x):
        h1 = self.conv1(x)
        h1 = self.bn1(h1)
        h1 = self.relu1(h1)
        h1 = self.conv2(h1)
        h1 = self.bn2(h1)
        h1 = self.relu2(h1)

        h = self.shortcut1(x)
        h = self.sbn1(h)
        h1 = h1 + h

        h2 = self.conv3(h1)
        h2 = self.bn3(h2)
        h2 = self.relu3(h2)
        h2 = self.conv4(h2)
        h2 = self.bn4(h2)
        h2 = self.relu4(h2)

        h = self.shortcut2(h1)
        h = self.sbn2(h)
        h2 = h2 + h

        h3 = self.conv5(h2)
        h3 = self.bn5(h3)
        h3 = self.relu5(h3)
        h3 = self.conv6(h3)
        h3 = self.bn6(h3)
        h3 = self.relu6(h3)

        h = self.shortcut3(h2)
        h = self.sbn3(h)
        h3 = h3 + h

        x = h3.view(h3.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        x = nn.LayerNorm(x.size())(x.cpu())
        x = x.cuda()
        # x = F.normalize(x.cuda(1))
        return x
#
print('==> Building model..')

net = ResNet(18).cuda()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

WD = 1e-3
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=0.001,weight_decay=WD)
# optimizer = AdamW(net.parameters(), lr=0.001, weight_decay=WD)
# optimizer = Adam_GC(net.parameters(), lr=0.001)
# optimizer = Adam_GCC(net.parameters(), lr=0.001,weight_decay=WD)
# optimizer = Adam_GC2(net.parameters(), lr=0.001, weight_decay=WD)
# optimizer = Adam_GCC2(net.parameters(), lr=0.001, weight_decay=WD)

# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


#
def flat(data):
    data = np.argmax(data, axis=1)
    return data


epoch_list = []
error_list = []


# Training

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    total = 0
    total = total
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        inputs = inputs.type(torch.FloatTensor)
        inputs, targets = inputs.cuda(), targets
        outputs = net(inputs)
        # targets=torch.max(targets, 1)[1]
        # print(targets)
        loss = criterion(outputs, torch.max(targets, 1)[1].long())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        # predicted = torch.max(predicted, 1)[1].cuda()
        targets = torch.max(targets, 1)[1].cuda()
        predicted = predicted
        taccuracy = (torch.sum(predicted == targets.long()).type(torch.FloatTensor) / targets.size(0)).cuda()
        # print(type(predicted),type(targets),predicted,targets,'type(predicted),type(targets)')
        # correct += predicted.eq(targets).sum().item()
        train_error = 1 - taccuracy.item()

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.type(torch.FloatTensor)
            inputs, targets = inputs.cuda(), targets
            outputs = net(inputs)
            # targets=torch.max(targets, 1)[1]
            # print(targets)
            loss = criterion(outputs, torch.max(targets, 1)[1].long())
            scheduler.step()
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            # predicted = torch.max(predicted, 1)[1].cuda()
            targets = torch.max(targets, 1)[1].cuda()
            taccuracy = (torch.sum(predicted == targets.long()).type(torch.FloatTensor) / targets.size(0)).cuda()
            # print(type(predicted),type(targets),predicted,targets,'type(predicted),type(targets)')
            # correct += predicted.eq(targets).sum().item()
            test_error = 1 - taccuracy.item()
            print('test:', taccuracy.item(), '||', test_error)
            epoch_list.append(epoch)
            # accuracy_list.append(taccuracy.item())
            error_list.append(test_error)


for epoch in range(start_epoch, start_epoch + 500):
    train(epoch)
    test(epoch)

for name, param in net.named_parameters():
    print(name, '      ', param)


model = ResNet(18)
stat(model, (30,113,1))
