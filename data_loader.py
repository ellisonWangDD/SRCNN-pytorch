import torch
from torch.utils.data import Dataset, DataLoader
from utils import make_input
from PIL import Image
# from scipy

class SRCNN_dataset(Dataset):
    """
    Create dataset for SRCNN

    Args:
        config: config to get dataset from utils
        transfrom: optional transform to be applied on a sample
    """
    def __init__(self, config):
        self.inputs, self.labels = make_input(config)
    
    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        label_sample = self.labels[idx]
        
        # transpose channel because
        # numpy image H x W x C
        # torch image C x H x W
        input_sample = input_sample.transpose(2, 0, 1)
        label_sample = label_sample.transpose(2, 0, 1)

        # Wrap with tensor
        input_sample, label_sample = torch.Tensor(input_sample), torch.Tensor(label_sample)
    
        return input_sample, label_sample



# class Trainset(Dataset):
#     def __init__(self, data_list, transform=None, loader=default_loader):
#         imgs = []#tmp variable don't use self to bind
#
#         for index, row in data_list.iterrows():
#             imgs.append((row['img_path'], row['label']))#add to list with tuple
#         self.imgs = imgs
#         self.transfom = transform
#         self.loader = loader
#     def __getitem__(self, item):
#         filename, label = self.imgs[item]
#         img = self.loader(filename)
#         if self.transfom:
#             img = self.transfom(img)#因为DatLoader是从getitem进行取元素的,返回的元素即为DataLoader取到的元素
#         return img, label
#     def __len__(self):
#         return len(self.imgs)
#
#
# def loader(img_path):
#     return Image.open(img_path)