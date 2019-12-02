import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import f1_score, accuracy_score
from scipy.fftpack import fft
import time

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

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

y0 = torch.from_numpy(m0['X100_DE_time'][:4096 * 25]).squeeze()
y1 = torch.from_numpy(m1['X108_DE_time'][:4096 * 25]).squeeze()
y2 = torch.from_numpy(m2['X121_DE_time'][:4096 * 25]).squeeze()
y3 = torch.from_numpy(m3['X133_DE_time'][:4096 * 25]).squeeze()
y4 = torch.from_numpy(m4['X172_DE_time'][:4096 * 25]).squeeze()
y5 = torch.from_numpy(m5['X188_DE_time'][:4096 * 25]).squeeze()
y6 = torch.from_numpy(m6['X200_DE_time'][:4096 * 25]).squeeze()
y7 = torch.from_numpy(m7['X212_DE_time'][:4096 * 25]).squeeze()
y8 = torch.from_numpy(m8['X225_DE_time'][:4096 * 25]).squeeze()
y9 = torch.from_numpy(m9['X237_DE_time'][:4096 * 25]).squeeze()
y10 = torch.from_numpy(m10['X281_DE_time'][:4096 * 25]).squeeze()
y11 = torch.from_numpy(m11['X285_DE_time'][:4096 * 25]).squeeze()
y12 = torch.from_numpy(m12['X297_DE_time'][:4096 * 25]).squeeze()
y13 = torch.from_numpy(m13['X277_DE_time'][:4096 * 25]).squeeze()
y14 = torch.from_numpy(m14['X289_DE_time'][:4096 * 25]).squeeze()
y15 = torch.from_numpy(m15['X312_DE_time'][:4096 * 25]).squeeze()
y16 = torch.from_numpy(m16['X273_DE_time'][:4096 * 25]).squeeze()
y17 = torch.from_numpy(m17['X293_DE_time'][:4096 * 25]).squeeze()
y18 = torch.from_numpy(m18['X318_DE_time'][:4096 * 25]).squeeze()

start = time.time()

# Hyper Parameters
EPOCH = 100
BATCH_SIZE = 64
LR = 0.005

X = np.empty(shape=[1000, 1024])
train_data = torch.stack((y0, y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15,
						  y16, y17, y18), 0)
train_data = train_data.reshape(1000, 2048).type(torch.FloatTensor)
train_data = train_data.numpy()

for i in range(1000):  # Fast Fourier transform
	Y = fft(train_data[i])
	train_data[i] = np.abs(Y)
	train_data[i] = (train_data[i] / len(train_data[i]))
	X[i] = train_data[i][:1024]

X = preprocessing.scale(X)  # Normalized
train_data = X

# for i in range(1200):
#     train_data[i] = 2*(train_data[i] - np.min(train_data[i]))/(np.max(train_data[i])-np.min(train_data[i]))-1
# print(train_data)
# train_data = torch.from_numpy(train_data).type(torch.FloatTensor)

L1 = torch.zeros(50)
L2 = torch.ones(50)
L3 = L2.add(1)
L4 = L3.add(1)
L5 = L4.add(1)
L6 = L5.add(1)
L7 = L6.add(1)
L8 = L7.add(1)
L9 = L8.add(1)
L10 = L9.add(1)

labels = torch.cat((L1, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L2, L3, L4, L5, L6, L7, L8, L9, L10), 0)
labels = labels.numpy()
x_train, x_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.3)
x_train = torch.from_numpy(x_train).type(torch.FloatTensor).to(device)
x_test = torch.from_numpy(x_test).type(torch.FloatTensor).to(device)
train_loader = Data.DataLoader(dataset=x_train, batch_size=BATCH_SIZE, shuffle=True)
np.save("label.npy", labels)

class AutoEncoder(nn.Module):
	def __init__(self):
		super(AutoEncoder, self).__init__()

		self.encoder = nn.Sequential(
			nn.Linear(1024, 512),
			nn.Tanh(),
			nn.Linear(512, 256),
			nn.Dropout(0.5),
			nn.Tanh(),
			nn.Linear(256, 128),
			nn.Dropout(0.5),
			nn.Tanh(),
			nn.Linear(128, 10),  # compress to 3 features which can be visualized in plt
		)
		self.decoder = nn.Sequential(
			nn.Linear(10, 128),
			nn.Tanh(),
			nn.Linear(128, 256),
			nn.Dropout(0.5),
			nn.Tanh(),
			nn.Linear(256, 512),
			nn.Dropout(0.5),
			nn.Tanh(),
			nn.Linear(512, 1024),
			nn.Tanh(),
		)

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return encoded, decoded

autoencoder = AutoEncoder().to(device)
autoencoder.train(mode=True)
loss_func1 = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR, weight_decay=1e-5)

# ----------------------------------------------------------------- print internal parameters
# for name, param in autoencoder.named_parameters():
#     if name in ['decoder.6.bias']:
#         print(param)
#     if param.requires_grad:
#         print(name)
# for i in autoencoder.named_parameters():
#     print(i)
# -----------------------------------------------------------------

for epoch in range(EPOCH):  # train
	for step, x in enumerate(train_loader):
		b_x = x.view(-1, 1024).to(device)  # batch x, shape (batch, 1024)
		b_y = x.view(-1, 1024).to(device)  # batch y, shape (batch, 1024)

		encoded, decoded = autoencoder(b_x)
		loss1 = loss_func1(decoded, b_y)  # mean square error

		optimizer.zero_grad()  # clear gradients for this training step
		loss1.backward()  # backpropagation, compute gradients
		optimizer.step()  # apply gradients

encoded_data, _ = autoencoder(x_train)
encoded_data = encoded_data.cpu().detach().numpy()
clf = SVC(C=10.23, kernel='rbf', gamma=0.001, decision_function_shape='ovr')
clf.fit(encoded_data, y_train)
fit_score = clf.score(encoded_data, y_train)
print('SVM fit-score = ' + str(fit_score))

autoencoder.eval()  # Test

encoded_data, _ = autoencoder(x_test)
encoded_data = encoded_data.cpu().detach().numpy()
pred_y = clf.predict(encoded_data)
accuracy = sum(pred_y == y_test) / len(x_test)
print('Test Accuracy = ' + str(accuracy))
# macro = f1_score(y_test, pred_y, average='macro')
# micro = f1_score(y_test, pred_y, average='micro')
# weighted = f1_score(y_test, pred_y, average='weighted')
# none = f1_score(y_test, pred_y, average=None)
# print(macro, micro, weighted, none)

end = time.time()
print('Time cost = ' + str(end - start))
