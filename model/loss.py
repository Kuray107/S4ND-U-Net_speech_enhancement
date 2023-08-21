import torch.nn as nn

def get_TF_domain_loss_function(loss_fn_name):
    if loss_fn_name == "TF-MSE":
        return nn.MSELoss()
    elif loss_fn_name == "TF-MAE":
        return nn.L1Loss()
    else:
        raise NotImplementedError(
            "Loss function: {} is not implemented!".format(loss_fn_name)
        )
