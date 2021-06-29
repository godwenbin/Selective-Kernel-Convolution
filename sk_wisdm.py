import torch
from torch import nn
import numpy as np
import torch.utils.data as Data
import os
import torch.backends.cudnn as cudnn
# from thop import profile
# from thop import clever_format
import torch.optim as optim
# from Adam_GC import *
from SENET import *
from torchstat import stat

os.environ['CUDA_VISIBLE_DEVICES']='1'

# parser = argparse.ArgumentParser(description='PyTorch Har Training')
# parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
# parser.add_argument('--resume', '-r', action='store_true',
#                     help='resume from checkpoint')
# args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


print('==> Preparing data..')

train_x = np.load('/mnt/ba3b04da-ce1b-4c21-ad1b-3aff7d337cdf/gaowenbin/Project/SKNet/HAR_Dataset/wisdm/x_train.npy')
shape = train_x.shape
train_x = torch.from_numpy(np.reshape(train_x.astype(np.float), [shape[0], 1, shape[1], shape[2]]))
train_x = train_x.type(torch.FloatTensor).cuda()

train_y = (np.load('/mnt/ba3b04da-ce1b-4c21-ad1b-3aff7d337cdf/gaowenbin/Project/SKNet/HAR_Dataset/wisdm/y_train.npy'))
train_y = torch.from_numpy(train_y)
train_y = train_y.type(torch.FloatTensor).cuda()


test_x = np.load('/mnt/ba3b04da-ce1b-4c21-ad1b-3aff7d337cdf/gaowenbin/Project/SKNet/HAR_Dataset/wisdm/x_test.npy')
test_x = torch.from_numpy(np.reshape(test_x.astype(np.float), [test_x.shape[0], 1, test_x.shape[1], test_x.shape[2]]))
test_x = test_x.type(torch.FloatTensor)

test_y = np.load('/mnt/ba3b04da-ce1b-4c21-ad1b-3aff7d337cdf/gaowenbin/Project/SKNet/HAR_Dataset/wisdm/y_test.npy')
test_y = torch.from_numpy(test_y.astype(np.float32))
test_y = test_y.type(torch.FloatTensor)


print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)
trainset = Data.TensorDataset(train_x, train_y)
trainloader = Data.DataLoader(dataset=trainset, batch_size=210, shuffle=True, num_workers=0)

testset = Data.TensorDataset(test_x, test_y)
testloader = Data.DataLoader(dataset=testset, batch_size=3295, shuffle=True, num_workers=0)


class SKConv(nn.Module):
    def __init__(self, features, M, G, r, stride, L):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = int(features / r)
        print(d)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=(3,1), stride=stride, padding=(1 + i,1), dilation=(1 + i,1), groups=G,
                          bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=(1,0), stride=(1,0), bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=False))
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, kernel_size=(1,0), stride=(1,0))
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        batch_size = x.shape[0]

        feats = [conv(x) for conv in self.convs]
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])

        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(feats * attention_vectors, dim=1)

        return feats_V


class SKNet(nn.Module):
    def __init__(self, M=3, G=32, r=32, stride=1, L=16):
        """ Constructor
        Args:
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            L: the minimum dim of the vector z in paper.
        """
        super(SKNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, (5,1), stride=(3,1),padding=(1,0), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2_sk = SKConv(128, M=M, G=G, r=r, stride=stride, L=L)

        self.conv3 = SKConv(256, M=M, G=G, r=r, stride=stride, L=L)
        # self.se= SELayer(128,16)

        self.fc = nn.Sequential(
            nn.Linear(59136, 6)
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2_sk(x)
        x = self.conv3(x)
        # x = self.se(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        x = nn.LayerNorm(x.size())(x.cpu())
        x = x.cuda()
        # print(x.shape)
        return x

# Model
print('==> Building model..')

net = SKNet().cuda()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

WD = 1e-3
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=0.001,weight_decay=WD)
# optimizer = AdamW(net.parameters(), lr=0.001, weight_decay=WD)
# optimizer = Adam_GC(net.parameters(), lr=0.001,weight_decay=WD)
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
    # torch.save(net,'/mnt/ba3b04da-ce1b-4c21-ad1b-3aff7d337cdf/gaowenbin/Project/SKNet/mobile/wisdm.pth')
# for name, param in net.named_parameters():
#     print(name, '      ', param)


model = SKNet()
stat(model, (1, 200, 3))
