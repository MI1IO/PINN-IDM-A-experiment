import numpy as np
import torch
from openpyxl.styles.builtins import output
from torch import nn
import pandas as pd
import torch.utils.data
import torch.utils.data as Data
import torch.nn.functional as F
import random
from sklearn import preprocessing
from matplotlib import pyplot as plt
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
DEVICE = torch.device("cuda")
SOS_value = 1.1
EOS_value = -1.1
PAD_value = 0

#位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # 创建全0张量pe
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 在1这个位置加一维
        div_term = 1 / (10000 ** ((2 * np.arange(d_model)) / d_model))
        # 通过sin和cos定义positional encoding
        pe[:, 0::2] = torch.sin(position * div_term[0::2])
        pe[:, 1::2] = torch.cos(position * div_term[1::2])
        pe = pe.unsqueeze(0).transpose(0, 1)  # [5000, 1, d_model],so need seq-len <= 5000
        self.register_buffer('pe', pe)  # 定义不可训练的模型参数

    def forward(self, x):
        # repeat的作用:对于repeat(x,y,z),把参数通道数复制x遍,行复制y遍,列复制z遍
        return x + self.pe[:x.size(0), :].repeat(1, x.shape[1], 1)  # 输入张量加上位置编码




# 无tensor梯度的IDM模型
def model_IDM(inputs_IDM, his_labels, output_length, s_0, T, a, b, v_d, dt):
    v_pred = torch.zeros((inputs_IDM.shape[0], output_length, 1))
    y_pred = torch.zeros((inputs_IDM.shape[0], output_length, 1))
    acc = torch.zeros((inputs_IDM.shape[0], output_length, 1))
    y = inputs_IDM[:, 0]
    v = inputs_IDM[:, 1]
    s = inputs_IDM[:, 2]
    delta_v = inputs_IDM[:, 3]

    s_x = s_0 + torch.max(torch.tensor(0), v * T + ((v * delta_v) / (2 * (a * b) ** 0.5)))
    # s_x = s_0 + torch.max(torch.zeros_like(v), v * T + ((v * delta_v) / (2 * (a * b) ** 0.5)))
    # s_x = s_0 + torch.max(torch.tensor(0),s_x = s_0 + torch.max(torch.zeros_like(v), v * T + ((v * delta_v) / (2 * (a * b) ** 0.5))) v * T + ((v * delta_v) / (2 * (a * b) ** 0.5)))
    # s_x = torch.tensor(2.5)+ torch.max(torch.tensor(0), v*torch.tensor(1.25)+((v*delta_v)/(2*(torch.tensor(1.75)*torch.tensor(1.25))**0.5)))
    a_f = a * (1 - (v / v_d) ** 4 - (s_x / s) ** 2)
    # a_f = torch.tensor(1.75)*(1-(v/torch.tensor(30))**4-(s_x/s)**2)
    v_pred[:, 0, 0] = v + a_f * dt
    for i in range(len(v_pred)):
        if v_pred[i, 0, 0] <= 0:
            v_pred[i, 0, 0] = 0
    y_pred[:, 0, 0] = y + v_pred[i, 0, 0] * dt
    acc[:, 0, 0] = a_f

    for i in range(y_pred.shape[0]):
        for j in range(output_length - 1):
            v = v_pred[i, j, 0]
            delta_v = his_labels[i, j, 1] - v_pred[i, j, 0]
            s = his_labels[i, j, 0] - y_pred[i, j, 0]
            # s_x = self.s_0 + self.T*v - ((v * delta_v)/(2*(self.a*self.b)**0.5))
            # s_x = s_0 +  v*T-((v*delta_v)/(2*(a*b)**0.5))
            s_x = s_0 + torch.max(torch.tensor(0), v * T + ((v * delta_v) / (2 * (a * b) ** 0.5)))
            # acc_temp = self.a*(1-(v/self.v_d)**4-(s_x/s)**2)
            acc_temp = a * (1 - (v / v_d) ** 4 - (s_x / s) ** 2)
            v2 = v + acc_temp * dt
            if v2 <= 0:
                v2 = 0
                acc_temp = (v2 - v) / dt
            y1 = y_pred[i, j, 0]
            y2 = y1 + v2 * dt
            acc[i, j + 1, 0] = acc_temp
            v_pred[i, j + 1, 0] = v2
            y_pred[i, j + 1, 0] = y2

    return y_pred



#PUNN部分 (Transformer)
class TransformerModel(nn.Module):

    def __init__(self, ninput, ntoken, ninp, nhead, nhid, fusion_size, nlayers, dropout, output_length, s_0, T, a, b, v_d, dt):
        super(TransformerModel, self).__init__()

        self.model_type = 'Transformer'
        self.src_mask = None
        self.embedding_layer = nn.Linear(ninput, ninp)
        self.encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(self.encoder_layers, nlayers)
        # self.relu = nn.ReLU()
        # self.leaky_relu = nn.LeakyReLU(0.1)
        # self.sig = nn.Sigmoid()
        self.decoder = nn.Linear(ninp, ntoken)
        self.dropout_in = nn.Dropout(dropout)
        self.fusion_layer_1 = nn.Linear(fusion_size, ntoken)
        self.fusion_layer_2 = nn.Linear(fusion_size, ntoken)
        self.output_length = output_length
        self.s_0 = s_0
        self.T = T
        self.a = a
        self.b = b
        self.v_d = v_d
        self.dt = dt

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 1
        # self.transformer_encoder.weight.data.uniform_(-initrange, initrange)
        self.embedding_layer.bias.data.zero_()
        self.embedding_layer.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, his_labels, s_0_input, T_input, a_input, b_input, v_d_input):
        #Transformer 模型输出
        a1=inputs.size()
        src_inputs = inputs[:, :, 0].unsqueeze(2)
        a2=src_inputs.size()
        src = src_inputs.transpose(1, 0)
        a3=src.size()
        src = self.embedding_layer(src)
        a4 = src.size()
        # pos_src = self.pos_encoder(src)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self._generate_square_subsequent_mask(len(src))
            self.src_mask = mask

        enc_src = self.transformer_encoder(src, self.src_mask)
        # a=enc_src.size()
        enc_src = enc_src[-1]
        # b=enc_src.size()
        enc_src = enc_src.repeat(self.output_length, 1, 1)
        c=enc_src.size()
        output = self.decoder(enc_src)
        d=output.size()
        output = output.transpose(0, 1)
        e=output.size()
        #历史信息
        dv = (src_inputs[:, -1, 0] - src_inputs[:, -(1+self.output_length), 0]) / self.output_length
        b1=dv.size()
        hist = torch.zeros(output.shape).to(DEVICE)
        b2=hist.size()
        for i in range(src_inputs.shape[0]):
            hist[i, :, 0] = torch.linspace(src_inputs[i, -1, 0].item(), src_inputs[i, -1, 0].item() + dv[i].item() * self.output_length,
                                        (self.output_length+1))[1:]

        #物理信息,使用训练过程中传入的IDM参数
        output_IDM = model_IDM(inputs[:, -1, :], his_labels[:, :, :], self.output_length, s_0_input, T_input, a_input, b_input, v_d_input, self.dt).to(DEVICE)

        # output_IDM = model_IDM(inputs[:, -1, :], his_labels[:, :, :], self.output_length, self.s_0, self.T, self.a, self.b, self.v_d, self.dt).to(DEVICE)
        #结果输出端融合
        # print(his_labels.device)
        # print(hist.device)
        # print(output_IDM.device)
        fusion = torch.cat([output, hist, output_IDM], axis=2)
        b3 = fusion.size()
        final_output = self.fusion_layer_1(fusion)
        b4 = final_output.size()
        return final_output



def idm_model_tensor(params, v, h, dv):
    """
    IDM模型张量实现
    参数：
    - params: [a, b, v0, s0, T]
    - v: 后车速度，张量
    - h: 间距，张量
    - dv: 速度差，张量
    返回：
    - a_calc: 当前加速度
    """
    # 使用模型参数
    # a, b, v0, s0, T = params
    s0, T, a, b, v0 = params

    # eps = 1e-6  # 避免除零
    eps = 0  # 避免除零
    s_star = s0 + torch.maximum(v * T + v * dv / (2 * torch.sqrt(a * b + eps)), torch.tensor(0.0))

    # s_star = s0 + max(v * T + v * dv / (2 * torch.sqrt(a * b + eps)), 0)
    # s_star = s0 + v * T + v * dv / (2 * torch.sqrt(a * b + eps))
    a_calc = a * (1 - (v / (v0 + eps)) ** 4 - (s_star / (h + eps)) ** 2)
    # if torch.any(a_calc < -1000):
    #     print("Warning: Some values in sv_v_next are less than -1000!")

    dt=0.1
    #检查是否有速度是负值，并对速度进行修正，但是这个代码未经验证
    v2 = v + a_calc * dt
    a51=v2.size()
    for i in range(len(v2)):
        if v2[i] <= 0:
            v2[i] = 0
            a_calc[i] = (v2[i] - v[i]) / dt

    if torch.any(a_calc < -1000):

        print("Warning: Some values in sv_v_next are less than -1000!")
    # return v2    #返回下一时刻的速度
    return a_calc   #返回加速度


# 模型的定义保持不变，只需确保 params 张量在初始化时有 requires_grad
class trajectory_predict_tensor(nn.Module):
    def __init__(self, params):
        super(trajectory_predict_tensor, self).__init__()
        self.params = nn.Parameter(params)  # IDM parameters

    def forward(self, sv_v1, h1, dv1, sv_Y1, lv_Y_seq1, lv_v_seq1):
        """
        张量实现的轨迹预测
        参数：
        - sv_v1, h1, dv1, sv_Y1: 当前时刻的后车速度、间距、速度差和位置
        - lv_Y_seq1, lv_v_seq1: 前车位置和速度序列
        返回：
        - result: 包含预测结果的字典
        """
        dt = 0.1
        time_steps = 30

        # 获取第 49 个时刻的数据
        sv_v = sv_v1[:, 49, :].unsqueeze(1)  # shape [batch_size, 1, 1]
        h = h1[:, 49, :].unsqueeze(1)        # shape [batch_size, 1, 1]
        dv = dv1[:, 49, :].unsqueeze(1)      # shape [batch_size, 1, 1]
        sv_Y = sv_Y1[:, 49, :].unsqueeze(1)  # shape [batch_size, 1, 1]
        lv_Y_seq = lv_Y_seq1[:, 50:80, :]    # shape [batch_size, 30, 1]
        lv_v_seq = lv_v_seq1[:, 50:80, :]   # shape [batch_size, 30, 1]

        # 初始化结果张量，按时间步展开
        sv_v_pred = sv_v
        h_pred = h
        dv_pred = dv
        sv_Y_pred = sv_Y

        for i in range(time_steps):
            #先计算a后计算v
            a_temp = idm_model_tensor(self.params, sv_v_pred[:, -1, :], h_pred[:, -1, :], dv_pred[:, -1, :])  # Use model parameters here
            sv_v_next = sv_v_pred[:, -1, :] + a_temp * dt
            # sv_v_next = idm_model_tensor(self.params, sv_v_pred[:, -1, :], h_pred[:, -1, :], dv_pred[:, -1, :])  # Use model parameters here

            delta_v_next = sv_v_next - lv_v_seq[:, i, :]
            sv_Y_next = sv_Y_pred[:, -1, :] + sv_v_next * dt
            distance_next = lv_Y_seq[:, i, :] - sv_Y_next

            if torch.any(sv_Y_next < -1000):
                print(sv_Y_next)
                print(a_temp)
                print("Warning: Some values in sv_v_next are less than -1000!")


            # 使用 torch.cat 拼接新的值
            sv_v_pred = torch.cat((sv_v_pred, sv_v_next.unsqueeze(1)), dim=1)
            h_pred = torch.cat((h_pred, distance_next.unsqueeze(1)), dim=1)
            dv_pred = torch.cat((dv_pred, delta_v_next.unsqueeze(1)), dim=1)
            sv_Y_pred = torch.cat((sv_Y_pred, sv_Y_next.unsqueeze(1)), dim=1)


        return sv_Y_pred[:, 1:, :]  # 去掉初始时刻的预测结果，返回预测位置


















# #PINN部分 (IDM)
# class IDMModel(nn.Module):
#     def __init__(self, s_0, T, a, b, v_d):
#         super(IDMModel, self).__init__()
#         self.model_type = 'IDM'
#         self.dt = 0.1
#         self.s_0 = torch.tensor([1.667], requires_grad=True)
#         self.T = torch.tensor([0.504], requires_grad=True)
#         self.a = torch.tensor([0.430], requires_grad=True)
#         self.b = torch.tensor([3.216], requires_grad=True)
#         self.v_d = torch.tensor([16.775], requires_grad=True)
#
#         self.s_0 = torch.nn.Parameter(self.s_0)
#         self.T = torch.nn.Parameter(self.T)
#         self.a = torch.nn.Parameter(self.a)
#         self.b = torch.nn.Parameter(self.b)
#         self.v_d = torch.nn.Parameter(self.v_d)
#
#         self.s_0.data.fill_(s_0)
#         self.T.data.fill_(T)
#         self.a.data.fill_(a)
#         self.b.data.fill_(b)
#         self.v_d.data.fill_(v_d)
#
#     # def forward(self, inputs_IDM, his_labels):
#     def forward(self, inputs_IDM):
#         x0=inputs_IDM.size()
#         y = inputs_IDM[:, -1,0]
#         x1=y.size()
#         v = inputs_IDM[:, -1, 1]
#         x2 = v.size()
#         s = inputs_IDM[:, -1, 2]
#         x3=s.size()
#         delta_v = inputs_IDM[:, -1, 3]
#         x4=delta_v.size()
#         s_x = self.s_0 + v * self.T + ((v * delta_v) / (2 * (self.a * self.b) ** 0.5))
#         a_f = self.a * (1 - (v / self.v_d) ** 4 - (s_x / s) ** 2)
#         v_pred = v + a_f * self.dt
#         for i in range(len(v_pred)):
#             if v_pred[i] <= 0:
#                 v_pred[i] == 0
#         x5=y.size()
#         x6=v_pred.size()
#         output_IDM = y + v_pred * self.dt
#
#         return output_IDM.unsqueeze(1).unsqueeze(2), torch.Tensor(self.s_0.data.cpu().numpy()), torch.Tensor(
#             self.T.data.cpu().numpy()), torch.Tensor(self.a.data.cpu().numpy()), torch.Tensor(
#             self.b.data.cpu().numpy()), torch.Tensor(self.v_d.data.cpu().numpy())



# # 开始训练
# def train():
#     # 定义早期停止的参数
#     best_loss = float('inf')  # 初始化为一个较大的值
#     early_stop_patience = 5000  # 当连续5个epoch损失没有减小时停止训练
#     early_stop_counter = 0  # 记录连续损失没有减小的次数
#
#     for epoch in range(EPOCH):
#         model1.train()
#         current_epoch_loss_PUNN = 0
#         current_epoch_loss_PINN = 0
#         batches_per_epoch = 0
#         # for X, Y in train_loader:
#         for batch in train_loader:
#             X, Y, Z, leader_y, leader_v = batch
#             a15 = X.size()
#
#
#             # a16=follower_v_batch.size()
#             Y_pred = model1(X[0:24,:,:], Y[0:24,:,:])
#             a13=Y_pred.size()
#
#             follower_v_batch=X[24:32,:,1].unsqueeze(2)
#             dis_batch=X[24:32,:,2].unsqueeze(2)
#             dv_batch = X[24:32, :, 3].unsqueeze(2)
#             follower_y = X[24:32, :, 0].unsqueeze(2)
#             Y_phy = model2(follower_v_batch, dis_batch, dv_batch, follower_y, leader_y[24:32, :, :], leader_v[24:32, :, :])  # Pass input data here
#             Y_PUNN2=model1(X[24:32,:,:], Y[24:32,:,:])
#             # a12 = Y_phy.size()
#             # Y_phy_all = model2(X)
#             # Y_phy=Y_phy_all[0]
#             #
#             # a12 = Y_phy.size()
#             # 通过损失函数和优化器来更新
#             updater1.zero_grad()
#             updater2.zero_grad()
#
#             loss_PUNN = loss_function(Y_pred, Z[0:24,:,:])
#             loss_PINN=  loss_function(Y_phy, Y_PUNN2)
#             # if loss_PINN<-1000 or loss_PINN>1000:
#             #     print("error")
#             # if torch.isnan(loss_PINN):
#             #     print("Loss is NaN!")
#             #     print(Y_phy)
#             #     print(Z[24:32,:,:])
#             #     print(Z)
#             #     break  # 停止训练，进行进一步排查
#
#             # loss = loss_function(Y_pred[:30, :, :], decoder_output[:30, :, :])  # 因为从倒数第PREDICT_SIZE个数到最后一个数是所有预测的原数值，能够进行比较
#             current_epoch_loss_PUNN += float(loss_PUNN) * BATCH_SIZE
#             current_epoch_loss_PINN += float(loss_PINN) * BATCH_SIZE
#
#             batches_per_epoch += BATCH_SIZE  # 这是一种loss的计算方法
#             loss_PUNN.backward()
#             loss_PINN.backward()
#             updater1.step()  # 梯度下降更新权重
#             updater2.step()  # 梯度下降更新权重
#
#             # sched.step()   #学习率折减
#
#         avg_loss_PUNN = current_epoch_loss_PUNN / batches_per_epoch
#         avg_loss_PINN = current_epoch_loss_PINN / batches_per_epoch
#
#         print(f"*Current epoch:{epoch} PUNN training loss MSE:{avg_loss_PUNN}")
#         print(f"*Current epoch:{epoch} PINN training loss MSE:{avg_loss_PINN}")
#
#         if epoch % 10 < 0.001:
#             val()
#
#         if epoch % 5 < 0.001:
#             torch.save(model1.state_dict(), "model state/state_{}".format(epoch))
#         avg_loss=0.7*avg_loss_PUNN+0.3*avg_loss_PINN
#         # 检查是否需要早期停止
#         if avg_loss < best_loss:
#             best_loss = avg_loss
#             early_stop_counter = 0
#         else:
#             early_stop_counter += 1
#
#         # 如果连续多个epoch损失没有减小，就停止训练
#         if early_stop_counter >= early_stop_patience:
#             print("Early stopping triggered! Training halted.")
#             break
#
#         # print(f"*Current epoch:{epoch} training loss:{current_epoch_loss / batches_per_epoch}")
#         torch.save(model1.state_dict(), "model state/state")