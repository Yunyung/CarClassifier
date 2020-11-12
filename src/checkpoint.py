
"""
Class Object used to save the state of trained NN model state
It also used for restart traininig process 
"""
class CheckPointStore():
    
    def __init__(self, total_epoch = 0, Best_val_Acc = 0.0, model_state_dict = None, epoch_loss_history = {}, epoch_acc_history = {}):
        self.total_epoch = total_epoch
        self.Best_val_Acc = Best_val_Acc
        self.model_state_dict = model_state_dict
        self.epoch_loss_history = epoch_loss_history
        self.epoch_acc_history = epoch_acc_history
