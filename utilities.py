# ------------------------------------------------------------------------------
# Utility functions that are used elsewhere but for readabilty reasons have been
# moved here.
# ------------------------------------------------------------------------------

import os
import torch


def load_model(model):
    """
    Load the model parameters from an existing trained model. 
    """
    fin = False
    backup1 = False

    if os.path.exists("TrainedModel/finalModel.pth"):
        fin = True
    elif os.path.exists("TrainedModel/modelBackup.pth"):
        backup1 = True

    if fin:
        try:
            model.load_state_dict(torch.load("TrainedModel/finalModel.pth"))
            return model
        except:
            print("finalModel seems to be corrupted, trying a backup...")
    
    if fin or backup1:
        try:
            model.load_state_dict(torch.load("TrainedModel/modelBackup.pth"))
            return model
        except:
            print("modelBackup seems to be corrupted, welp")

    print("There doesn't seem to be anything to load.")
    return model
