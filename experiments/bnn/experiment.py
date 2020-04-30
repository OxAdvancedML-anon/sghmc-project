from update import SGD, SGDMomentum, SGLD, SGHMC
from sklearn.utils import shuffle
import torch
import torch.nn as nn

from misc import get_dataset, set_seed
from model import gen_model
from train import Trainer
from test import Tester
from log import Logger


num_train, num_test, batch_size = 50000, 10000, 500
model_dims = (28 * 28, 100, 10)
epochs, num_burn, num_runs = 50, 10, 1
precision_params = (alpha, beta) = 1.0, 1.0
average_results = True
seed = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

updaters = {
    "SGD": SGD(device, lr=0.16e-4 * num_train, num_train=num_train),
    "SGD with momentum": SGDMomentum(device, lr=0.01e-5 * num_train, momentum_coeff=0.01, num_train=num_train),
    "SGLD": SGLD(device, lr=0.8e-4 * num_train, num_train=num_train, precision_params=precision_params),  # best with 0.8e-4
    "SGHMC": SGHMC(device, lr=0.2e-5 * num_train, momentum_coeff=0.01, num_train=num_train, precision_params=precision_params),  # normally at 0.2
}

set_seed(seed)
x_train_, y_train_ = get_dataset(batch_size, num_train, train=True)
x_test, y_test = get_dataset(batch_size, num_test, train=False)
x_test, y_test = torch.from_numpy(x_test).to(device), torch.from_numpy(y_test).to(device)
loss_fn = nn.NLLLoss()
logger = Logger(epochs, num_burn)


for updater_name, updater in updaters.items():
    logger.new_updater(updater_name)
    for run in range(1, num_runs + 1):
        logger.new_run()
        seed += 1
        set_seed(seed)
        model = gen_model(*model_dims).to(device)
        updater.set_weights(model.parameters())
        x_train_, y_train_ = shuffle(x_train_, y_train_, random_state=seed)
        x_train, y_train = torch.from_numpy(x_train_).to(device), torch.from_numpy(y_train_).to(device)

        trainer = Trainer(model, loss_fn, updater)
        tester = Tester(model, num_burn, average_results)

        for epoch in range(1, epochs + 1):
            trainer.run(x_train, y_train)
            nll, err = tester.run(x_test, y_test, epoch)
            logger.log(run, epoch, nll, err)

    logger.plot()
logger.show_plot()
