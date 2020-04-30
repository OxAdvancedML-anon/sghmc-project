import numpy as np


class Tester:
    def __init__(self, model, num_burn, take_average):
        self.model = model
        self.o_pred = np.zeros((20, 500, 10), 'float32')
        self.num_burn = num_burn
        self.take_average = take_average
        self.y_pred_average = None

    def run(self, x_test, y_test, epoch):
        alpha = self.get_weight_for_average(epoch)
        incorrect = np.empty(0)
        loss = np.empty(0)

        for i, (x, y) in enumerate(zip(x_test, y_test)):
            model_out = self.model(x).detach().cpu().numpy()
            self.o_pred[i, :] = alpha * model_out + (1.0 - alpha) * self.o_pred[i, :]
            y_pred = np.argmax(self.o_pred[i, :], 1)
            incorrect = np.concatenate([incorrect, y_pred != y.cpu().numpy()])
            for j in range(20):
                loss = np.append(loss, self.o_pred[i, j, y[j]])

        return -loss.mean(), incorrect.mean()

    def get_weight_for_average(self, epoch):
        if self.take_average and epoch > self.num_burn:
            return 1 / (epoch - self.num_burn + 1)
        else:
            return 1
