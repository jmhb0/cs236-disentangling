import torch
import torchvision
import numpy as np 
from abc import ABC, abstractmethod

DEFAULT_DATASETS_DIR="/pasteur/u/jmhb/vae-transform-invariance/data/datasets"

from abc import ABC, abstractmethod
import os

class BaseTensorDatasetClass(torch.utils.data.Dataset, ABC):
    """
    Base class for handling very simple datasets where the data and labels are
    tensors.
    Implementing classes must define __init__() and import self.data and
    self.labels, otherwise an error will be thrown (which is handled by the
    inherited ABC class).
    """
    @property
    @abstractmethod
    def __init__(self, dir_dataset=None, transform=None, train=True, flatten=False, binarize=False):
        self.data, self.labels = None, None
        self.transform = transform
    
    def post_process_data(self, **kwargs):
        """To be executed after child classes have saved self.data and 
        self.labels. It must be explicitly called at the end of __init__() 
        in child implementations. 
        """
        self.img_shape_original = self.data.shape[1:]
        self.img_shape = self.data.shape[1:]
        self.m, self.n = self.img_shape_original[-2:]
        # for grayscale, where image is 2-dim, add a 3rd channel dimension
        if len(self.img_shape_original)==2:
            self.img_shape_original = (1,*self.img_shape_original)
        # we flatten the data in the __getitem__ instead of before
        self.do_flatten  = kwargs.get("flatten", False)
        #if kwargs.get("flatten", False): 
        #    self.flatten(); self.do_flatten=True
        # else: self.do_flatten=False
        if kwargs.get("binarize", False): self.binarize()
        if kwargs.get("norm_0_1", False): self.norm_0_1()
        device = kwargs.get("device", 'cuda') 
        print(f"Putting dataset on {device}")
        self.data, self.labels = self.data.to(device), self.labels.to(device)

    def save_dataset(self, save_dir):
        print("Saving")
        raise NotImplementedError()

    def binarize(self):
        """ (most naive) For all data, set values>0 to 1. Convert to int."""
        self.data[self.data>0] = 1
        self.data[self.data<=0] = 0
        #self.data=self.data.to(dtype=torch.uint8)
    
    def flatten(self):
        self.flatten=True
        l = len(self.data)
        self.data = self.data.reshape(l, -1).contiguous()

    def norm_0_1(self):
        l, u = self.data.min(), self.data.max()
        self.data = (self.data-l) / (u-l) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, indx):
        is_single_indx = type(indx) is int
        if is_single_indx: indx=slice(indx,indx+1) # so transforms always work

        data, label = self.data[indx], self.labels[indx]

        if self.transform: 
            data = self.transform(data)
        if self.do_flatten: 
            data = data.view(data.size(0), -1)
        if is_single_indx: 
            data, label = data[0], label[0]

        return data, label

class RotatedMnistBepler(BaseTensorDatasetClass):
    """
    From http://bergerlab-downloads.csail.mit.edu/spatial-vae/mnist_rotated.tar.gz
    """
    def __init__(self, dir_dataset=f"{DEFAULT_DATASETS_DIR}/mnist_rotated", 
            train=True, norm_0_1=True,
            **kwargs):
        if train:
            self.data = np.load(f"{dir_dataset}/images_train.npy")
            self.labels = np.load(f"{dir_dataset}/transforms_train.npy")
        else:
            self.data = np.load(f"{dir_dataset}/images_test.npy")
            self.labels = np.load(f"{dir_dataset}/transforms_test.npy")

        self.data, self.labels = torch.Tensor(self.data), torch.Tensor(self.labels)
        assert len(self.data)==len(self.labels)
        kwargs['norm_0_1']=norm_0_1
        self.post_process_data(**kwargs)

class RotatedTranslatedMnistBepler(BaseTensorDatasetClass):
    """
    From http://bergerlab-downloads.csail.mit.edu/spatial-vae/mnist_rotated_translated.tar.gz
    """
    def __init__(self, dir_dataset=f"{DEFAULT_DATASETS_DIR}/mnist_translated",
                 train=True,
                 **kwargs):
        if train:
            self.data = np.load(f"{dir_dataset}/images_train.npy")
            self.labels = np.load(f"{dir_dataset}/transforms_train.npy")
        else:
            self.data = np.load(f"{dir_dataset}/images_test.npy")
            self.labels = np.load(f"{dir_dataset}/transforms_test.npy")

        self.data, self.labels = torch.Tensor(self.data), torch.Tensor(self.labels)
        assert len(self.data)==len(self.labels)
        self.post_process_data(**kwargs)

class AllenMitoObjects(BaseTensorDatasetClass):
    """
    From https://web.stanford.edu/~jmhb/datasets/allen_mito_objects.tar.gz
    """
    def __init__(self, dir_dataset=f"{DEFAULT_DATASETS_DIR}/allen_mito_objects", 
            train=True,
            dim=32,
            **kwargs):
        self.dim=dim
        if train:
            if dim==36: self.data = torch.load(f"{dir_dataset}/allen_mito_objects_36_train.sav")
            elif dim==32: self.data = torch.load(f"{dir_dataset}/allen_mito_objects_32_train.sav")
            else: self.data = torch.load(f"{dir_dataset}/allen_mito_objects_train.sav")
            self.labels = torch.load(f"{dir_dataset}/allen_mito_labels_train.sav")
        else:
            if dim==36: self.data = torch.load(f"{dir_dataset}/allen_mito_objects_36_test.sav")
            elif dim==32: self.data = torch.load(f"{dir_dataset}/allen_mito_objects_32_test.sav")
            else: self.data = torch.load(f"{dir_dataset}/allen_mito_objects_test.sav")
            self.labels = torch.load(f"{dir_dataset}/allen_mito_labels_test.sav")
            
        # unsqueeze the channel dimension
        self.data = self.data.unsqueeze(1)

        self.transform=kwargs.get('transform',None)
        self.post_process_data(**kwargs)

    """
    def  __getitem__(self, indx):
        data, label = self.data[indx], self.labels[indx]
        m=n=self.dim
        # this unfortunate bit of code is to distinguish the cases where get() is run for 
        # a single value (so there is no batch dimension), and also where we have flattened 
        # the input array, rather than keeping it in full image form. 
        # In the first branch
        if (data.ndim==1 and self.do_flatten) or (data.ndim==2 and not self.do_flatten):
            if self.transform:
                data = self.transform(data.view(1,m,n))
            if self.do_flatten: 
                data = data.view(-1)
            else: 
                data = data.view(1,m,n)
        else: 
            b = len(data)
            if self.transform:
                data = self.transform(data.view(b,m,n))
            if self.do_flatten: 
                data = data.view(b,-1)
            else:
                data = data.view(b,1,m,n)
        return data, label
        """


class MNIST(BaseTensorDatasetClass):
    """
    Use the torchvision dataset source. Reformat it so that it's held in a
    single tensor to be consistent with everything else.
    """
    def __init__(self, dir_dataset=f"{DEFAULT_DATASETS_DIR}/mnist", train=True,
            **kwargs):
        super(BaseTensorDatasetClass)
        fname_modifier = 'train' if train else 'test'
        fname_data=f"{dir_dataset}/mnist_x_{fname_modifier}_tensor.sav"
        fname_label=f"{dir_dataset}/mnist_y_{fname_modifier}_tensor.sav"

        if not os.path.isfile(fname_data) or not os.path.isfile(fname_label):
            self.get_mnist_dataset_from_torchvision(train, dir_dataset, fname_data, fname_label)

        self.data = torch.load(fname_data)
        self.labels = torch.load(fname_label)

        self.post_process_data(**kwargs)

    def get_mnist_dataset_from_torchvision(self, train, dir_dataset, fname_data, fname_label):
        """
        Get MNIST from torchvision package, put in single-tensor dataset form,
        and save to a `.sav` file to be later read by __init__() method.
        """
        print("Downloading MNIST from torchvision source")
        x = torchvision.datasets.MNIST(root=dir_dataset, download=True, train=train,
                transform=torchvision.transforms.ToTensor())
        x_shape = x[0][0].shape
        x_len = len(x)
        x_tensor = torch.Tensor(x_len, *x_shape)
        y_tensor = torch.Tensor(x_len, 1)
        for i in range(x_len):
            x_tensor[i], y_tensor[i] = x[i][0], x[i][1]
        torch.save(x_tensor, fname_data)
        torch.save(y_tensor, fname_label)
