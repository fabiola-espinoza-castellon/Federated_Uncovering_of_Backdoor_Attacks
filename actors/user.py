class User():
    def __init__(self, idx, train_indexes, test_indexes):
        self.idx = idx
        self.train_indexes = train_indexes
        self.test_indexes = test_indexes
        self.attacked = False
        self.nk = len(train_indexes)
        self.clean_acc = []
        self.poise_acc = []
