from scipy.io import loadmat
import scipy.io as scio
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import pywt
import pywt.data

plt.rcParams['font.sans-serif'] = ['SimHei']  # For plot with Chinese
plt.rcParams['axes.unicode_minus'] = False

m0 = loadmat("data/正常轴承数据/100.mat")
m1 = loadmat("data/12k 驱动端故障轴承数据/108.mat")
m2 = loadmat("data/12k 驱动端故障轴承数据/121.mat")
m3 = loadmat("data/12k 驱动端故障轴承数据/133.mat")
m4 = loadmat("data/12k 驱动端故障轴承数据/172.mat")
m5 = loadmat("data/12k 驱动端故障轴承数据/188.mat")
m6 = loadmat("data/12k 驱动端故障轴承数据/200.mat")
m7 = loadmat("data/12k 驱动端故障轴承数据/212.mat")
m8 = loadmat("data/12k 驱动端故障轴承数据/225.mat")
m9 = loadmat("data/12k 驱动端故障轴承数据/237.mat")
m10 = loadmat("data/12k 扇端故障轴承数据/281.mat")
m11 = loadmat("data/12k 扇端故障轴承数据/285.mat")
m12 = loadmat("data/12k 扇端故障轴承数据/297.mat")
m13 = loadmat("data/12k 扇端故障轴承数据/277.mat")
m14 = loadmat("data/12k 扇端故障轴承数据/289.mat")
m15 = loadmat("data/12k 扇端故障轴承数据/312.mat")
m16 = loadmat("data/12k 扇端故障轴承数据/273.mat")
m17 = loadmat("data/12k 扇端故障轴承数据/293.mat")
m18 = loadmat("data/12k 扇端故障轴承数据/318.mat")

y1 = (m0['X100_DE_time'])
y2 = (m1['X108_DE_time'])
y3 = (m2['X121_DE_time'])
y4 = (m3['X133_DE_time'])
y5 = (m4['X172_DE_time'])
y6 = (m5['X188_DE_time'])
y7 = (m6['X200_DE_time'])
y8 = (m7['X212_DE_time'])
y9 = (m8['X225_DE_time'])
y10 = (m9['X237_DE_time'])
y11 = (m10['X281_DE_time'])
y12 = (m11['X285_DE_time'])
y13 = (m12['X297_DE_time'])
y14 = (m13['X277_DE_time'])
y15 = (m14['X289_DE_time'])
y16 = (m15['X312_DE_time'])
y17 = (m16['X273_DE_time'])
y18 = (m17['X293_DE_time'])
y19 = (m18['X318_DE_time'])
y1 = torch.from_numpy(y1[:4096 * 25]).squeeze()
y2 = torch.from_numpy(y2[:4096 * 25]).squeeze()
y3 = torch.from_numpy(y3[:4096 * 25]).squeeze()
y4 = torch.from_numpy(y4[:4096 * 25]).squeeze()
y5 = torch.from_numpy(y5[:4096 * 25]).squeeze()
y6 = torch.from_numpy(y6[:4096 * 25]).squeeze()
y7 = torch.from_numpy(y7[:4096 * 25]).squeeze()
y8 = torch.from_numpy(y8[:4096 * 25]).squeeze()
y9 = torch.from_numpy(y9[:4096 * 25]).squeeze()
y10 = torch.from_numpy(y10[:4096 * 25]).squeeze()
y11 = torch.from_numpy(y11[:4096 * 25]).squeeze()
y12 = torch.from_numpy(y12[:4096 * 25]).squeeze()
y13 = torch.from_numpy(y13[:4096 * 25]).squeeze()
y14 = torch.from_numpy(y14[:4096 * 25]).squeeze()
y15 = torch.from_numpy(y15[:4096 * 25]).squeeze()
y16 = torch.from_numpy(y16[:4096 * 25]).squeeze()
y17 = torch.from_numpy(y17[:4096 * 25]).squeeze()
y18 = torch.from_numpy(y18[:4096 * 25]).squeeze()
y19 = torch.from_numpy(y19[:4096 * 25]).squeeze()

y = torch.stack((y1, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19), 0)
y = y.reshape(-1, 4096)
y = y.numpy()

mode = pywt.Modes.periodic  # mode

def plot_signal_decomposition(data, w, title):
    w = pywt.Wavelet(w)  # Pick wavelet function
    a = data
    ca = []
    cd = []
    for i in range(3):
        (a, d) = pywt.dwt(a, w, mode)  # Perform a 3rd order discrete wavelet package transform
    ca.append(a)
    cd.append(d)

    a = data
    (a, d) = pywt.dwt(a, w, mode)
    (a, d) = pywt.dwt(a, w, mode)
    (a, d) = pywt.dwt(d, w, mode)
    ca.append(a)
    cd.append(d)

    a = data
    (a, d) = pywt.dwt(a, w, mode)
    (a, d) = pywt.dwt(d, w, mode)
    (a, d) = pywt.dwt(a, w, mode)
    ca.append(a)
    cd.append(d)

    a = data
    (a, d) = pywt.dwt(a, w, mode)
    (a, d) = pywt.dwt(d, w, mode)
    (a, d) = pywt.dwt(d, w, mode)
    ca.append(a)
    cd.append(d)

    # print(ca[0].shape)  518
    # ----------------------------------------------------------------------- plot
    #     x = np.linspace(0, 518, 518)
    #     fig = plt.figure()
    #
    #     plt.subplot(811)
    #     plt.plot(x, ca[0], color='blue', linewidth=1.0, linestyle='-')
    #     #plt.xlim((0, 0.35))
    #     # plt.ylim((-0.25, 0.25))
    #     #plt.yticks(np.linspace(-0.25, 0.25, 3))
    #     #plt.xlabel('time(t/s)')
    #     plt.ylabel('FC1')
    #     #plt.title(u'正常状态', fontsize='medium', fontweight='bold')
    #
    #     plt.subplot(813)
    #     plt.plot(x, ca[1], color='blue', linewidth=1.0, linestyle='-')
    #     plt.ylabel('FC3')
    #
    #     plt.subplot(815)
    #     plt.plot(x, ca[2], color='blue', linewidth=1.0, linestyle='-')
    #     plt.ylabel('FC5')
    #
    #     plt.subplot(817)
    #     plt.plot(x, ca[3], color='blue', linewidth=1.0, linestyle='-')
    #     plt.ylabel('FC7')
    #
    #     plt.subplot(812)
    #     plt.plot(x, cd[0], color='blue', linewidth=1.0, linestyle='-')
    #     plt.ylabel('FC2')
    #
    #     plt.subplot(814)
    #     plt.plot(x, cd[1], color='blue', linewidth=1.0, linestyle='-')
    #     plt.ylabel('FC4')
    #
    #     plt.subplot(816)
    #     plt.plot(x, cd[2], color='blue', linewidth=1.0, linestyle='-')
    #     plt.ylabel('FC6')
    #
    #     plt.subplot(818)
    #     plt.plot(x, cd[3], color='blue', linewidth=1.0, linestyle='-')
    #     plt.ylabel('FC8')
    #
    #     fig.tight_layout()
    #     plt.show()
    # -----------------------------------------------------------------------

    An = []
    Dn = []
    c = []
    sum1 = 0
    sum2 = 0
    for i in range(4):
        for j in range(len(ca[i])):
            sum2 = sum2 + math.pow(cd[i][j], 2)
            sum1 = sum1 + math.pow(ca[i][j], 2)
        An.append(sum1)
        Dn.append(sum2)
        sum2 = 0
        sum1 = 0
    c = An + Dn
    return c


X = []
for i in range(500):
    Y = y[i]
    c = plot_signal_decomposition(Y, 'db4', "DWT")
    X.append(c)
dataNew = 'all.mat'
scio.savemat(dataNew, {'all_DE_time': X})  # pack all the transformed data

# plt.show()
