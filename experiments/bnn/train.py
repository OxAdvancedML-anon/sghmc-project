class Trainer:
    def __init__(self, model, loss_fn, updater):
        self.model = model
        self.loss_fn = loss_fn
        self.updater = updater

    def run(self, x_train, y_train):
        for x, y in zip(x_train, y_train):
            self.updater.zero_grad()
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)
            loss.backward()
            self.updater.run()
        if self.updater.gibbs_step:
            self.updater.run_gibbs_step()
