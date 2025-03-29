import numpy as np
import torch
from torch import nn
import pandas as pd
import torch.utils.data
import torch.utils.data as Data
import torch.nn.functional as F
import random
from sklearn import preprocessing
from matplotlib import pyplot as plt
import time

start_time = time.time()
from sklearn.preprocessing import MinMaxScaler
import openpyxl
from utils import PositionalEncoding, TransformerModel, trajectory_predict_tensor

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(42)
ninput = 1
ntoken = 1
ninp = 14
nhead = 2
nhid = 28
fusion_size = 3
nlayers = 3
dropout = 0.1
output_length = 30
dt = 0.1
lr_PUNN = 0.0005
lr_PINN = 0.0000001
# epoch_num = 10
alpha = 0.7
WINDOW_SIZE = 50
PREDICT_SIZE = 30
LABEL_SIZE = WINDOW_SIZE
BATCH_SIZE = 32
BATCH_SIZE_test=1
EPOCH = 100




DEVICE = torch.device("cuda")

data_raw = pd.read_csv('CF_pair_80.csv').values
# data_raw = pd.read_csv('CF_pair_80_1.csv').values
# data_raw = pd.read_csv('CF_pair_80_40.csv').values
# data_raw = pd.read_csv('CF_pair_10.csv').values


# 提取所需列
sv_y_v = data_raw[:, [2]]
distance = data_raw[:, [8]]
delta_v = data_raw[:, [9]]
sv_Local_Y = data_raw[:, [1]]
lv_Local_Y = data_raw[:, [5]]
lv_y_v = data_raw[:, [6]]

#归一化
input_x = data_raw[:,[2,8,9,1]]
max_num = np.max(input_x[:,3])+185
min_num = np.min(input_x[:,3])
# max_num = np.max(data_raw[:,[2,8,9,1,5,6]])
# min_num = np.min(data_raw[:,[2,8,9,1,5,6]])

# #数据归一化的方式有待商榷
# sv_y_v=(sv_y_v-min_num)/(max_num-min_num)
# distance=(distance-min_num)/(max_num-min_num)
# delta_v=(delta_v-min_num)/(max_num-min_num)
# sv_Local_Y=(sv_Local_Y-min_num)/(max_num-min_num)
# lv_Local_Y=(lv_Local_Y-min_num)/(max_num-min_num)
# lv_y_v=(lv_y_v-min_num)/(max_num-min_num)
sv_y_v=sv_y_v/(max_num-min_num)
distance=distance/(max_num-min_num)
delta_v=delta_v/(max_num-min_num)
sv_Local_Y=sv_Local_Y/(max_num-min_num)
lv_Local_Y=lv_Local_Y/(max_num-min_num)
lv_y_v=lv_y_v/(max_num-min_num)
# 数据分割与准备
follower_v, dis, dv, follower_y, leader_y, leader_v = [], [], [], [], [], []



s_0 = 1.667/(max_num-min_num)
# T = 0.504/(max_num-min_num)
T = 0.504    #这个是源码里面的设置
a = 0.430/(max_num-min_num)
b = 3.216/(max_num-min_num)
v_d = 16.775/(max_num-min_num)




num=0
for i in range(0, len(data_raw), 80):  # 每15秒产生150条数据
    follower_v.append(sv_y_v[i:i + WINDOW_SIZE + PREDICT_SIZE])
    dis.append(distance[i:i + WINDOW_SIZE + PREDICT_SIZE])
    dv.append(delta_v[i:i + WINDOW_SIZE + PREDICT_SIZE])
    follower_y.append(sv_Local_Y[i:i + WINDOW_SIZE + PREDICT_SIZE])
    leader_y.append(lv_Local_Y[i:i + WINDOW_SIZE + PREDICT_SIZE])
    leader_v.append(lv_y_v[i:i + WINDOW_SIZE + PREDICT_SIZE])
    num=num+1

print(num)

# 打乱数据
indices = list(range(len(follower_v)))
random.shuffle(indices)
follower_v = [follower_v[i] for i in indices]
dis = [dis[i] for i in indices]
dv = [dv[i] for i in indices]
follower_y = [follower_y[i] for i in indices]
leader_y = [leader_y[i] for i in indices]
leader_v = [leader_v[i] for i in indices]

# 转化为Tensor
follower_v = torch.tensor(np.array(follower_v), dtype=torch.float32)
dis = torch.tensor(np.array(dis), dtype=torch.float32)
dv = torch.tensor(np.array(dv), dtype=torch.float32)
follower_y = torch.tensor(np.array(follower_y), dtype=torch.float32)
leader_y = torch.tensor(np.array(leader_y), dtype=torch.float32)
leader_v = torch.tensor(np.array(leader_v), dtype=torch.float32)

# 划分数据集
num = len(follower_v)
train_boundary = int(num * 0.8)
val_boundary = int(num * 0.9)

train_follower_v = follower_v[:train_boundary, :].to(DEVICE)
train_dis = dis[:train_boundary, :].to(DEVICE)
train_dv = dv[:train_boundary, :].to(DEVICE)
train_follower_y = follower_y[:train_boundary, :].to(DEVICE)
train_leader_y = leader_y[:train_boundary, :].to(DEVICE)
train_leader_v = leader_v[:train_boundary, :].to(DEVICE)

val_follower_v = follower_v[train_boundary:val_boundary, :].to(DEVICE)
val_dis = dis[train_boundary:val_boundary, :].to(DEVICE)
val_dv = dv[train_boundary:val_boundary, :].to(DEVICE)
val_follower_y = follower_y[train_boundary:val_boundary, :].to(DEVICE)
val_leader_y = leader_y[train_boundary:val_boundary, :].to(DEVICE)
val_leader_v = leader_v[train_boundary:val_boundary, :].to(DEVICE)

test_follower_v = follower_v[val_boundary:, :].to(DEVICE)
test_dis = dis[val_boundary:, :].to(DEVICE)
test_dv = dv[val_boundary:, :].to(DEVICE)
test_follower_y = follower_y[val_boundary:, :].to(DEVICE)
test_leader_y = leader_y[val_boundary:, :].to(DEVICE)
test_leader_v = leader_v[val_boundary:, :].to(DEVICE)


# 修改test_follower_v中每组第51到第80条数据等于第50条的数据
for i in range(0, len(train_leader_v)):  # 每80条数据为一组
    # 取得第50条数据
    value_to_set1 = train_leader_v[i][49]
    # 设置第51到第80条数据等于第50条数据
    train_leader_v[i][50:80] = value_to_set1
    value_to_set2 = train_follower_v[i][50]    # 取得第51条数据的加速度
    for k in range(30):
        train_leader_y[i][k+50] = train_leader_y[i][k+49]+value_to_set2*0.1

# x=train_leader_y.size()

for i in range(0, len(val_leader_v)):  # 每80条数据为一组
    # 取得第50条数据
    value_to_set1 = val_leader_v[i][49]
    # 设置第51到第80条数据等于第50条数据
    val_leader_v[i][50:80] = value_to_set1
    value_to_set2 = val_follower_v[i][50]    # 取得第51条数据的加速度
    for k in range(30):
        val_leader_y[i][k+50] = val_leader_y[i][k+49]+value_to_set2*0.1

# 修改test_follower_v中每组第51到第80条数据等于第50条的数据
for i in range(0, len(test_leader_v)):  # 每80条数据为一组
    # 取得第50条数据
    value_to_set1 = test_leader_v[i][49]
    # 设置第51到第80条数据等于第50条数据
    test_leader_v[i][50:80] = value_to_set1
    value_to_set2 = test_follower_v[i][50]    # 取得第51条数据的加速度
    for k in range(30):
        test_leader_y[i][k+50] = test_leader_y[i][k+49]+value_to_set2*0.1



#input_x和input_hist
#input_x [follower_y, follower_v, dis, dv]
#input_hist[leader_y, leader_v]
train_input_x=torch.cat((train_follower_y[:,0:50,:], train_follower_v[:,0:50,:], train_dis[:,0:50,:], train_dv[:,0:50,:]), dim=-1)
train_input_hist=torch.cat((train_leader_y[:,50:80,:], train_leader_v[:,50:80,:]), dim=-1)

val_input_x=torch.cat((val_follower_y[:,0:50,:], val_follower_v[:,0:50,:], val_dis[:,0:50,:], val_dv[:,0:50,:]), dim=-1)
val_input_hist=torch.cat((val_leader_y[:,50:80,:], val_leader_v[:,50:80,:]), dim=-1)

test_input_x=torch.cat((test_follower_y[:,0:50,:], test_follower_v[:,0:50,:], test_dis[:,0:50,:], test_dv[:,0:50,:]), dim=-1)
test_input_hist=torch.cat((test_leader_y[:,50:80,:], test_leader_v[:,50:80,:]), dim=-1)


# 使用DataLoader加载数据
train_dataset = Data.TensorDataset(train_input_x, train_input_hist,train_follower_y[:,50:80,:], train_leader_y, train_leader_v)   #这里做了修改
# train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)  #样本量不足batchsize时会丢弃
val_dataset = Data.TensorDataset(val_input_x, val_input_hist,val_follower_y[:,50:80,:], val_leader_y, val_leader_v)
# val_loader = Data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = Data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
test_dataset = Data.TensorDataset(test_input_x, test_input_hist, test_follower_y[:,50:80,:])
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE_test, shuffle=False)






model1 = TransformerModel(ninput, ntoken, ninp, nhead, nhid, fusion_size, nlayers, dropout, output_length, s_0, T, a, b, v_d, dt)  # 维度为7，应该是总共7种数据类型
model1 = model1.to(DEVICE)  # cuda训练
from torch.nn.parallel import DataParallel

# model = DataParallel(model)
# 定义损失函数和优化器等
loss_function = nn.MSELoss()
loss_function1 = nn.L1Loss()
# loss_function = nn.SmoothL1Loss()

updater1 = torch.optim.Adam(model1.parameters(), lr_PUNN)
# sched1 = torch.optim.lr_scheduler.StepLR(updater1, step_size=15, gamma=0.1)

params = torch.tensor([s_0, T, a, b, v_d], dtype=torch.float32, requires_grad=True, device=DEVICE)  # IDM模型的参数
model2 = trajectory_predict_tensor(params=params).to(DEVICE)
model2 = model2.to(DEVICE)
updater2 = torch.optim.Adam(model2.parameters(), lr_PINN)

# s_0_t = 1.667 / (max_num - min_num)
# # T = 0.504/(max_num-min_num)
# T_t = 0.504  # 这个是源码里面的设置
# a_t = 0.430 / (max_num - min_num)
# b_t = 3.216 / (max_num - min_num)
# v_d_t = 16.775 / (max_num - min_num)



#
def val():
    model1.eval()  # 设置模型为评估模式
    test_loss = 0.0
    # correct = 0
    nu = 0
    val_current_epoch_loss_PUNN=0
    val_current_epoch_loss_PINN=0
    with torch.no_grad():  # 禁用梯度计算
        current_epoch_loss = 0
        current_epoch_loss1 = 0
        batches_per_epoch = 0

        # for X, Y in val_loader:
        for name, param in model2.named_parameters():
            # print(f"Epoch {epoch}, Model2 Parameter {name}: {param.data}")
            s_0_t = param.data[0].item()
            T_t = param.data[1].item()
            a_t = param.data[2].item()
            b_t = param.data[3].item()
            v_d_t = param.data[4].item()
        for batch in val_loader:
            X, Y, Z, leader_y, leader_v = batch
            #model1
            Y_pred = model1(X[0:24, :, :], Y[0:24, :, :], s_0_t, T_t, a_t, b_t, v_d_t)  # 前向传播
            loss_PUNN = loss_function(Y_pred, Z[0:24, :, :])  # 计算PUNN损失
            val_current_epoch_loss_PUNN += float(loss_PUNN) * BATCH_SIZE
            #model2
            follower_v_batch = X[24:32, :, 1].unsqueeze(2)
            dis_batch = X[24:32, :, 2].unsqueeze(2)
            dv_batch = X[24:32, :, 3].unsqueeze(2)
            follower_y = X[24:32, :, 0].unsqueeze(2)
            Y_phy = model2(follower_v_batch, dis_batch, dv_batch, follower_y, leader_y[24:32, :, :],
                           leader_v[24:32, :, :])
            Y_PUNN2 = model1(X[24:32, :, :], Y[24:32, :, :], s_0_t, T_t, a_t, b_t, v_d_t)
            loss_PINN = loss_function(Y_phy, Y_PUNN2)  # 计算PINN损失

            val_current_epoch_loss_PINN += float(loss_PINN) * BATCH_SIZE

            batches_per_epoch += BATCH_SIZE

        val_avg_loss_PUNN = val_current_epoch_loss_PUNN / batches_per_epoch
        val_avg_loss_PINN = val_current_epoch_loss_PINN / batches_per_epoch
        val_avg_loss=alpha*val_avg_loss_PUNN+(1-alpha)*val_avg_loss_PINN

        print(f"*PUNN val loss MSE:{val_avg_loss_PUNN}")
        print(f"*PINN val loss MSE:{val_avg_loss_PINN}")
    return val_avg_loss


train_input_x=torch.cat((train_follower_y[:,0:50,:], train_follower_v[:,0:50,:], train_dis[:,0:50,:], train_dv[:,0:50,:]), dim=-1)



def train():
    best_train_loss = float('inf')  # 初始化为一个较大的值
    best_val_loss = float('inf')
    early_stop_patience = 10  # 当连续指定次epoch损失没有减小时停止训练
    early_stop_counter = 0  # 记录连续损失没有减小的次数

    for epoch in range(EPOCH):
        model1.train()
        model2.train()
        train_current_epoch_loss_PUNN = 0
        train_current_epoch_loss_PINN = 0
        batches_per_epoch = 0
        for name, param in model2.named_parameters():
            formatted_params = [f"{parameter.item():.7f}" for parameter in param.data.flatten()]
            print(f"*Current epoch: {epoch}, Model2 Parameter {name}: {', '.join(formatted_params)}")

        for batch in train_loader:
            # Print model2 parameters (weights and biases)
            # for name, param in model2.named_parameters():
            #     formatted_params = [f"{val.item():.6f}" for val in param.data.flatten()]
            #     print(f"*Current epoch: {epoch}, Model2 Parameter {name}: {', '.join(formatted_params)}")
            for name, param in model2.named_parameters():
                # print(f"Epoch {epoch}, Model2 Parameter {name}: {param.data}")
                s_0_t = param.data[0].item()
                T_t = param.data[1].item()
                a_t = param.data[2].item()
                b_t = param.data[3].item()
                v_d_t = param.data[4].item()
            X, Y, Z, leader_y, leader_v = batch

            # --- Model1 Forward and Backward ---
            updater1.zero_grad()  # 清空model1梯度
            Y_pred = model1(X[0:24, :, :], Y[0:24, :, :], s_0_t, T_t, a_t, b_t, v_d_t)  # 前向传播
            loss_PUNN = loss_function(Y_pred, Z[0:24, :, :])  # 计算PUNN损失
            loss_PUNN.backward()  # 反向传播
            updater1.step()  # 更新model1参数
            train_current_epoch_loss_PUNN += float(loss_PUNN) * BATCH_SIZE

            # --- Model2 Forward and Backward ---
            updater2.zero_grad()  # 清空model2梯度
            follower_v_batch = X[24:32, :, 1].unsqueeze(2)
            dis_batch = X[24:32, :, 2].unsqueeze(2)
            dv_batch = X[24:32, :, 3].unsqueeze(2)
            follower_y = X[24:32, :, 0].unsqueeze(2)
            Y_phy = model2(follower_v_batch, dis_batch, dv_batch, follower_y, leader_y[24:32, :, :], leader_v[24:32, :, :])
            Y_PUNN2 = model1(X[24:32, :, :], Y[24:32, :, :], s_0_t, T_t, a_t, b_t, v_d_t)
            loss_PINN = loss_function(Y_phy, Y_PUNN2)  # 计算PINN损失
            loss_PINN.backward()  # 反向传播
            updater2.step()  # 更新model2参数
            train_current_epoch_loss_PINN += float(loss_PINN) * BATCH_SIZE

            batches_per_epoch += BATCH_SIZE

        train_avg_loss_PUNN = train_current_epoch_loss_PUNN / batches_per_epoch
        train_avg_loss_PINN = train_current_epoch_loss_PINN / batches_per_epoch
        train_avg_loss=alpha*train_avg_loss_PUNN+(1-alpha)*train_avg_loss_PINN

        print(f"*Current epoch:{epoch} PUNN training loss MSE:{train_avg_loss_PUNN}")
        print(f"*Current epoch:{epoch} PINN training loss MSE:{train_avg_loss_PINN}")
        print(f"*Current epoch:{epoch} Average training loss MSE:{train_avg_loss}")
        # for name, param in model2.named_parameters():
        #     formatted_params = [f"{parameter.item():.7f}" for parameter in param.data.flatten()]
        #     print(f"*Current epoch: {epoch}, Model2 Parameter {name}: {', '.join(formatted_params)}")
        if epoch % 1 < 0.001:
            val_avg_loss=val()
            print(f"*Current epoch:{epoch} Average validate loss MSE:{val_avg_loss}")
            if val_avg_loss<best_val_loss:
                best_val_loss=val_avg_loss
                torch.save(model1.state_dict(), "model state/best_val_state_{}".format(epoch))
                torch.save(model2.state_dict(), "model state/best_val_state_phy{}".format(epoch))
        if epoch % 5 < 0.001:
            torch.save(model1.state_dict(), "model state/state_{}".format(epoch))
            torch.save(model2.state_dict(), "model state/state_phy{}".format(epoch))
        train_avg_loss=0.7*train_avg_loss_PUNN+0.3*train_avg_loss_PINN
        # 检查是否需要早期停止
        if train_avg_loss < best_train_loss:
            best_train_loss = train_avg_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # 如果连续多个epoch损失没有减小，就停止训练
        if early_stop_counter >= early_stop_patience:
            print("Early stopping triggered! Training halted.")
            break
        torch.save(model1.state_dict(), "model state/state")
# loss_infer = []



def infer():
    model1.eval()  # 设置模型为评估模式
    BATCH_SIZE_INFER = 1
    test_loss = 0.0
    nu = 0
    predict_size=30
    # 初始化 ADE、FDE 和 RFDE 累计值
    total_ADE = 0.0
    total_FDE = 0.0
    total_RFDE = 0.0
    RFDE_11=0.0
    RFDE_22 = 0.0
    total_samples = 0
    for name, param in model2.named_parameters():
        # print(f"Epoch {epoch}, Model2 Parameter {name}: {param.data}")
        s_0_t = param.data[0].item()
        T_t = param.data[1].item()
        a_t = param.data[2].item()
        b_t = param.data[3].item()
        v_d_t = param.data[4].item()
    with torch.no_grad():  # 禁用梯度计算
        current_epoch_loss = 0
        current_epoch_loss1 = 0
        batches_per_epoch = 0

        # for X, Y in test_loader:
        for batch in test_loader:
            X, Y, Z = batch

            Y_pred = model1(X, Y, s_0_t, T_t, a_t, b_t, v_d_t)


            predict = Y_pred.reshape(predict_size, 1)
            real = Z.reshape(predict_size, 1)

            # predict = predict * (max_num - min_num)+min_num
            # real = real * (max_num - min_num)+min_num
            predict = predict * (max_num - min_num)
            real = real * (max_num - min_num)

            inverse_predict = torch.tensor(np.array(predict.cpu().numpy()), dtype=torch.float32)
            inverse_real = torch.tensor(np.array(real.cpu().numpy()), dtype=torch.float32)
            # # 将 GPU 上的张量移到 CPU，再转换为 NumPy 数组
            # inverse_predict = torch.tensor(predict.cpu().numpy(), dtype=torch.float32)
            # inverse_real = torch.tensor(real.cpu().numpy(), dtype=torch.float32)
            predict=predict.cpu().numpy()
            real=real.cpu().numpy()

            # Loss 计算
            loss = loss_function(inverse_predict, inverse_real)
            current_epoch_loss += float(loss) * BATCH_SIZE
            # loss1 = loss_function1(inverse_predict, inverse_real)
            # current_epoch_loss1 += float(loss1) * BATCH_SIZE

            batches_per_epoch += BATCH_SIZE

            # 计算 ADE
            ADE = np.mean(np.sqrt((predict - real) ** 2))
            total_ADE += ADE

            # 计算 FDE
            FDE = np.sqrt((predict[-1] - real[-1]) ** 2)
            total_FDE += FDE

            # 计算 RFDE
            RFDE_1=np.abs(real[-1] - predict[-1])
            RFDE_2 = np.abs(real[-1] - real[0])

            RFDE_11 += RFDE_1
            RFDE_22 += RFDE_2


            total_samples += 1

        # 平均损失
        avg_loss = current_epoch_loss / batches_per_epoch
        # avg_loss1 = current_epoch_loss1 / batches_per_epoch

        # 平均指标
        avg_ADE = total_ADE / total_samples
        avg_FDE = total_FDE / total_samples
        avg_RFDE = RFDE_11 *100/ RFDE_22

        print(f"* infer_MSE test loss: {avg_loss}")
        # print(f"* infer_MAE test loss: {avg_loss1}")
        print(f"* ADE: {avg_ADE}")
        print(f"* FDE: {avg_FDE}")
        print(f"* RFDE: {avg_RFDE}%")


# MODEL_SAVED = True
MODEL_SAVED = False


if MODEL_SAVED:
    model1.load_state_dict(torch.load("model state/state_60"))
else:
    train()

# val()

infer()






end_time = time.time()
print('cost %f second' % (end_time - start_time))