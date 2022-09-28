from typing import Tuple
import numpy as np
import struct
import os
import time


class FullyConnectedLayer(object):
    def __init__(self, num_input, num_output):
        self.num_input = num_input
        self.num_output = num_output
        self.init_param()
        print('\tFully connected layer with input %d, output %d.' % (self.num_input, self.num_output))
    def init_param(self, std=0.01):
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.num_input, self.num_output))
        self.bias = np.zeros([1, self.num_output])
    def forward(self, input):
        start_time = time.time()
        self.input = input
        self.output = np.matmul(input, self.weight) + self.bias
        return self.output
    def backward(self, top_diff):
        self.d_weight = np.dot(self.input.T, top_diff)
        self.d_bias = np.sum(top_diff, axis=0)
        bottom_diff = np.dot(top_diff, self.weight.T)
        return bottom_diff
    def update_param(self, lr):
        self.weight = self.weight - lr*self.d_weight
        self.bias = self.bias - lr*self.d_bias
    def load_param(self, weight, bias):
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias
    def save_param(self):
        return self.weight, self.bias


class MSELossLayer(object):
    def __init__(self):
        print('\tMSE loss layer.')

    def get_loss(self, y_pred, y):
        self.x = y_pred
        self.y = y
        self.batch_size = self.x.shape[0]
        loss = np.sum(np.power((self.x - self.y), 2))/self.batch_size
        return loss

    def backward(self):
        bottom_diff = 2 *(self.x - self.y) / self.batch_size
        return bottom_diff


class Model:
    # Modify your model, default is a linear regression model with random weights
    ID_DICT = {"NAME": "Wenxin Zu", "BU_ID": "U75249292", "BU_EMAIL": "zcynthia@bu.edu"}

    def __init__(self, batch_size=128, input_size=764, hidden1=256, hidden2=128, hidden3=64, output_size=1, lr=0.01,
                 max_epoch=50, print_iter=100):
        self.theta = None
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.output_size = output_size
        self.lr = lr
        self.max_epoch = max_epoch
        self.print_iter = print_iter
        self.build_model()

    def shuffle_data(self, X: np.array, y: np.array) -> Tuple[np.array, np.array]:
        train_data = np.append(X, y, axis=1)
        np.random.shuffle(train_data)
        return train_data

    def preprocess(self, X: np.array, y: np.array, split: str = "train") -> Tuple[np.array, np.array]:
        ###############################################
        ####      add preprocessing code here      ####
        ###############################################
        if split == 'train':
            train_data = self.shuffle_data(X, y)
            X = train_data[:, : -1]
            y = train_data[:, -1].reshape(-1, 1)
        return X, y

    def build_model(self):

        self.fc1 = FullyConnectedLayer(self.input_size, self.hidden1)
        self.fc2 = FullyConnectedLayer(self.hidden1, self.hidden2)
        self.fc3 = FullyConnectedLayer(self.hidden2, self.hidden3)
        self.fc4 = FullyConnectedLayer(self.hidden3, self.output_size)
        self.mse = MSELossLayer()
        self.update_layer_list = [self.fc1, self.fc2, self.fc3, self.fc4]

    def init_model(self):

        for layer in self.update_layer_list:
            layer.init_param()

    def load_model(self, param_dir):

        params = np.load(param_dir).item()
        self.fc1.load_param(params['w1'], params['b1'])
        self.fc2.load_param(params['w2'], params['b2'])
        self.fc3.load_param(params['w3'], params['b3'])
        self.fc4.load_param(params['w4'], params['b4'])

    def save_model(self, param_dir):

        params = {}
        params['w1'], params['b1'] = self.fc1.save_param()
        params['w2'], params['b2'] = self.fc2.save_param()
        params['w3'], params['b3'] = self.fc3.save_param()
        params['w4'], params['b4'] = self.fc4.save_param()
        np.save(param_dir, params)

    def forward(self, input):
        h1 = self.fc1.forward(input)
        h2 = self.fc2.forward(h1)
        h3 = self.fc3.forward(h2)
        prob = self.fc4.forward(h3)
        return prob

    def backward(self):
        dloss = self.mse.backward()
        dh4 = self.fc4.backward(dloss)
        dh3 = self.fc3.backward(dh4)
        dh2 = self.fc2.backward(dh3)
        dh1 = self.fc1.backward(dh2)

    def update(self, lr):
        for layer in self.update_layer_list:
            layer.update_param(lr)

    def train(self, X_train: np.array, y_train: np.array):
        max_batch = X_train.shape[0] / self.batch_size

        for idx_epoch in range(self.max_epoch):
            for idx_batch in range(int(max_batch)):
                batch_X = X_train[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size]
                batch_Y = y_train[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size]
                prob = self.forward(batch_X)
                loss = self.mse.get_loss(prob, batch_Y)
                self.backward()
                self.update(self.lr)


    def predict(self, X_val: np.array) -> np.array:
        predict = self.forward(X_val)
        return predict
