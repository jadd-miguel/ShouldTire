class LoaderTrio:
    def __init__(self, tr_loader, val_loader, ts_loader):
        self.tr_loader = tr_loader
        self.val_loader = val_loader
        self.ts_loader = ts_loader

    def get_tr_loader(self): return self.tr_loader
    def get_val_loader(self): return self.val_loader
    def get_ts_loader(self): return self.ts_loader

class TrainingHistory:
    def __init__(self, output, tr_loss, tr_acc, val_loss, val_acc):
        self.output = output
        self.tr_loss = tr_loss
        self.tr_acc = tr_acc
        self.val_loss = val_loss
        self.val_acc = val_acc

    def get_output(self): return self.output
    def get_tr_loss(self): return self.tr_loss
    def get_tr_acc(self): return self.tr_acc
    def get_val_loss(self): return self.val_loss
    def get_val_acc(self): return self.val_acc
