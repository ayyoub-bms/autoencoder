import copy


class EarlyStopping:

    def __init__(self, max_iter=50):
        self.iter = 0
        self.stop = False
        self.max_iter = max_iter
        self.best_score = -1
        self.best_model = None

    def __call__(self, model, validation_loss):

        if self.best_score < 0:
            self.best_score = validation_loss
            self.best_model = copy.deepcopy(model)

        elif self.best_score <= validation_loss:
            self.iter += 1
            if self.iter >= self.max_iter:
                self.stop = True
        else:
            self.best_score = validation_loss
            self.best_model = copy.deepcopy(model)
            self.iter = 0
