#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import numpy as np
import os
import sys
import torch.nn as nn
from helper_code import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import torchvision.transforms as transforms
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    full_dataset = ECGDataset(data_path=os.path.join(data_folder))
    
    train_ratio = 0.8  # 80% 作为训练集
    val_ratio = 1 - train_ratio  # 20% 作为验证集

    # 计算训练集和验证集的大小
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # 随机划分数据集
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    train_records = [full_dataset.records[i] for i in train_subset.indices]
    val_records = [full_dataset.records[i] for i in val_subset.indices]

    train_dataset = ECGDataset(data_path=data_folder, records=train_records, oversample_factor=20)  # 只对训练集过采样
    val_dataset = ECGDataset(data_path=data_folder, records=val_records, oversample_factor=1)  # 验证集不变


    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    print(len(train_loader), len(test_loader))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter('./log/')

    # model = Transformer1d(
    #         n_classes=2, 
    #         n_length=1000, 
    #         d_model=12, 
    #         nhead=1, 
    #         dim_feedforward=128, 
    #         dropout=0.1, 
    #         activation='relu',
    #         verbose=True
    #     ).to(device)
        
    model = ResNet1D(
        in_channels=12, 
        base_filters=128, # 64 for ResNet1D, 352 for ResNeXt1D
        kernel_size=16, 
        stride=2, 
        groups=32, 
        n_block=48, 
        n_classes=2, 
        downsample_gap=6, 
        increasefilter_gap=12, 
        use_do=True)
    model.to(device)  

    # Create Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)


    if verbose:
        print('Training the model on the data...')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # train model
    step = 0       
    best_score = 0
    epoch = 0
    from tqdm import tqdm
    for _ in tqdm(range(10), desc="epoch", leave=False):
        model.train()
        prog_iter = tqdm(train_loader, desc="Training", leave=False)
        for i, (signals, labels) in enumerate(prog_iter):
                signals, labels = signals.to(device), labels.to(device)
                # print(labels)
                # if labels.sum() == 0:
                #     # print(f"Skipping batch {i}, no positive samples.")
                #     continue
                outputs = model(signals)
                # print(labels.shape, outputs.shape)
                loss1 = criterion(outputs, labels)
                # loss2 = auc_loss(outputs, labels)
                # print(loss2)
                loss = loss1 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
                writer.add_scalar('Loss/train_CELOSS', loss1.item(), step)
                # writer.add_scalar('Loss/train_Aucloss', loss2, step)
                # writer.add_scalar('Loss/train', loss, step)
                # print(f"Epoch [{epoch+1}/300], Iteration {i}, Loss: {loss.item():.4f}")
            
        scheduler.step(_) 
        model.eval()
        all_pred_prob = []
        all_gt = []
        with torch.no_grad():
            prog_iter = tqdm(test_loader, desc="Testing", leave=False)
            for i, (signals, labels) in enumerate(prog_iter):
                    signals, labels = signals.to(device), labels.to(device)
                    outputs = model(signals)
                    prob_class_1 = torch.sigmoid(outputs)[:, 1].cpu().numpy() 
                    all_pred_prob.append(prob_class_1)
                    all_gt.append(labels.cpu().numpy()) 
        all_pred_prob = np.concatenate(all_pred_prob)
        all_pred = (all_pred_prob > 0.5).astype(int)
        all_gt = np.concatenate(all_gt)
        
        challenge_score = compute_challenge_score(all_gt, all_pred_prob)
        auroc, auprc = compute_auc(all_gt, all_pred_prob)
        accuracy = compute_accuracy(all_gt, all_pred)
        f_measure = compute_f_measure(all_gt, all_pred)

        writer.add_scalar('challenge_score', challenge_score, _)
        writer.add_scalar('auroc', auroc, _)
        writer.add_scalar('auprc', auprc, _)
        writer.add_scalar('accuracy', accuracy, _)
        writer.add_scalar('f_measure', f_measure, _)
        # 保存最佳模型
        if challenge_score > best_score:
            save_model(model_folder, model)
            best_score = challenge_score
        epoch += 1


    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet1D(
        in_channels=12, 
        base_filters=128, # 64 for ResNet1D, 352 for ResNeXt1D
        kernel_size=16, 
        stride=2, 
        groups=32, 
        n_block=48, 
        n_classes=2, 
        downsample_gap=6, 
        increasefilter_gap=12, 
        use_do=True)
    # model.to(device) 
    best_model_path = os.path.join(model_folder, "best.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    # Load the model.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Extract the features.
    features = extract_features(record)
    # print(features.shape)
    signal = torch.tensor(features, dtype=torch.float32).to(device)
    signal = signal.T.unsqueeze(0)
    # print(signal.shape)
    # features = features.reshape(1, -1)
    outputs = model(signal)
    # Get the model outputs.
    
    probability_output = torch.sigmoid(outputs)[:, 1].cpu().detach().numpy() 
    if probability_output > 0.5:
        binary_output = 1 
    else:
        binary_output = 0

    return binary_output, probability_output

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################
class ECGDataset(Dataset):
    def __init__(self, data_path, oversample_factor=1, transform=None, records=None):
        super().__init__()
        self.data_path = data_path
        if records:
            self.records = records
        else:
            self.records = find_records(data_path)
        self.transform = transform


        # 加载所有样本的标签
        labels = [load_label(os.path.join(data_path, r)) for r in self.records]
        
        # 找到正负样本
        self.positive_records = [r for r, l in zip(self.records, labels) if l == 1]
        self.negative_records = [r for r, l in zip(self.records, labels) if l == 0]

        # 对正样本进行过采样
        self.positive_records = self.positive_records * oversample_factor  # 复制20倍
        self.records = self.negative_records + self.positive_records  # 组合新的数据集
        np.random.shuffle(self.records)  # 打乱顺序，防止过采样影响模型训练
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, index):
        record = os.path.join(self.data_path, self.records[index])
        signal = extract_features(record)
        label = load_label(record)

        # 转换为 PyTorch Tensor
        signal = torch.tensor(signal, dtype=torch.float32)  # [L, C]
        label = torch.tensor(label, dtype=torch.long)

        signal = signal.T
        if self.transform:
            signal = self.transform(signal)

        # return signal, demography, label
        return [signal, label]

class Transformer1d(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples, n_classes)
        
    Pararmetes:
        
    """
    def __init__(self, n_classes, n_length, d_model, nhead, dim_feedforward, dropout, activation, verbose=False):
        super(Transformer1d, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.n_length = n_length
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.n_classes = n_classes
        self.verbose = verbose
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.nhead, 
            dim_feedforward=self.dim_feedforward, 
            dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.dense = nn.Linear(self.d_model, self.n_classes)
        # self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        out = x
        out = out.permute(2, 0, 1)
        out = self.transformer_encoder(out)
        out = out.mean(0)
        out = self.dense(out)
        # out = self.softmax(out)    
        return out    
# Extract your features.

    
class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.conv(net)

        return net
        
class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.max_pool(net)
        
        return net
    
class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, is_first_block=False):
        super(BasicBlock, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            groups=self.groups)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            groups=self.groups)
                
        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        
        identity = x
        
        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        
        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)
            
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        
        # shortcut
        out += identity

        return out
    
class ResNet1D(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes
        
    """

    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, downsample_gap=2, increasefilter_gap=4, use_bn=True, use_do=True, verbose=False):
        super(ResNet1D, self).__init__()
        
        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do

        self.downsample_gap = downsample_gap # 2 for base model
        self.increasefilter_gap = increasefilter_gap # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters
                
        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            
            tmp_block = BasicBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride = self.stride, 
                groups = self.groups, 
                downsample=downsample, 
                use_bn = self.use_bn, 
                use_do = self.use_do, 
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        # self.do = nn.Dropout(p=0.5)
        self.dense = nn.Linear(out_channels, n_classes)
        # self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        out = x
        
        # first conv
        if self.verbose:
            print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        
        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        if self.verbose:
            print('final pooling', out.shape)
        # out = self.do(out)
        out = self.dense(out)
        if self.verbose:
            print('dense', out.shape)
        # out = self.softmax(out)
        if self.verbose:
            print('softmax', out.shape)
        
        return out   


from sklearn.metrics import roc_auc_score
def auc_loss(y_pred, y_true):
    y_true = y_true.detach().cpu().numpy()
    # print(y_true)
    y_pred = torch.sigmoid(y_pred)[:, 1].detach().cpu().numpy()
    # print(y_true)
    auc_loss_value = roc_auc_score(y_true, y_pred)
    
    # 取反后，损失越小表示性能越好
    return 1 - auc_loss_value


# # 示例数据
# y_true = torch.tensor([0, 1, 1, 0], dtype=torch.long)  # 真实标签
# y_pred = torch.tensor([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6], [0.1, 0.9]], dtype=torch.float32)  # 模型预测得分，假设第二列是正类的得分

# # 实例化损失函数并计算损失
# loss_fn = CombinedLoss(auc_weight=1.0)  # auc_weight 用来调整 AUC 损失的权重
# loss = loss_fn(y_true, y_pred)
# print(f"Combined Loss: {loss.item()}")


def extract_features(record):
    header = load_header(record)
    signal, fields = load_signals(record)
    L, C = signal.shape
    original_indices = np.linspace(0, 1, L)  # 归一化原始索引
    target_indices = np.linspace(0, 1, 5000)  # 归一化目标索引

    # **Step 2: 对每个通道进行插值**
    interpolated_signal = np.zeros((5000, C))  # 预分配新信号
    for i in range(C):
        f = interp1d(original_indices, signal[:, i], kind='linear', fill_value="extrapolate")  
        interpolated_signal[:, i] = f(target_indices)
    return interpolated_signal

# Save your trained model.
def save_model(model_folder, model):
    best_model_path = os.path.join(model_folder, "best.pth")
    torch.save(model.state_dict(), best_model_path)