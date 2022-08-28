import torch
from .pix2pixHD_model import Pix2PixHDModel

def create_model():
    model = Pix2PixHDModel()
    model.initialize()
    model = torch.nn.DataParallel(model, device_ids=[0])

    return model
