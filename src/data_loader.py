import math 
import random 
import numpy as np

import torch 
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset 

from config import get_config
config, unparsed = get_config() 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

class SiameseNetworkDataset(Dataset):
    def __init__(self, lst, dataset, should_invert=False):
        self.dataset = dataset
        self.should_invert = should_invert

        self.indices = lst
        self.labels = [dataset[idx][1] for idx in lst]      

        self.label_to_indices = {}
        self.indices_to_indices = {}
        for idx, label in zip(self.indices, self.labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
        
    def __getitem__(self,idx):
        index1 = self.indices[idx]
        label1 = self.labels[idx]

        #we need to make sure approx 50% of images are in the same class
        same_class_flag = random.randint(0,1) 

        if not same_class_flag:
            label2 = label1
            index2 = random.choice(self.label_to_indices[label1])
            while index1 == index2:
                index2 = random.choice(self.label_to_indices[label1])            
        else:
            possible_values = [i for i in range(10) if i != label1]
            label2 = random.choice(possible_values)
            index2 = random.choice(self.label_to_indices[label2])  

        img0, _ = self.dataset[index1]
        img1, _ = self.dataset[index2]
        
        # if self.should_invert:
        #     img0 = PIL.ImageOps.invert(img0)
        #     img1 = PIL.ImageOps.invert(img1)
        
        return img0, img1, torch.from_numpy(np.array(same_class_flag, dtype=np.float32)), torch.from_numpy(np.array([label1, label2], dtype=np.float32))
    
    def __len__(self):
        return len(self.indices)


def get_cifar10_train_loader(data_dir, batch_size, random_seed, shuffle=True, num_workers=4, pin_memory=True, model_num=5, split="homogeneous"): 
    trans = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize([0.4915, 0.4823, .4468], [0.2470, 0.2435, 0.2616])
    ])

    dataset = datasets.CIFAR10(root=data_dir,
                               transform=trans,
                               download=True,
                               train=True)
    set_seed(41)
    # if shuffle:
    #     np.random.seed(random_seed)
    # torch.manual_seed(random_seed)
    pnumber = model_num 

    lst = []
    class_size = len(dataset) // len(set(dataset.targets))
    num_classes = len(set(dataset.targets)) 

    # dictionary of labels map and initializing variables 
    labels = dataset.targets
    dct = {label: [] for label in set(labels)}
    for idx, label in enumerate(labels):
        dct[label].append(idx)

    for i in range(num_classes):
        temp = random.sample(dct[i], len(dct[i]))
        dct[i] = temp 

    # probabilities 
    torch.set_printoptions(precision=3)
    probs = []
    for i in range(num_classes):
        if split == 'homogeneous': 
            probs.append([1.0 / pnumber] * pnumber)
        elif split == 'imbalanced':
            rho = config.rho 
            n = config.alloc_n 
            major_allocation = [rho] * n 
            remaining_allocation = [(1 - sum(major_allocation)) / (model_num - n)] * (model_num - n) 
            prob = major_allocation + remaining_allocation 
            probs.append(prob) 
        else:  # heterogeneous 
            if pnumber == 2: 
                if i < 5:
                    probs.append([0.75, 0.25])
                else:
                    probs.append([0.25, 0.75])
            elif pnumber == 4: 
                if i == 0:
                    probs.append([0.97, 0.01, 0.01, 0.01])
                else:
                    probs.append([0.25, 0.25, 0.25, 0.25]) 


    print(probs, end="\n\n") 
    
    lst = {i: [] for i in range(pnumber)} 
    for class_id, distribution in enumerate(probs):
        from_id = 0 
        for participant_id, prob in enumerate(distribution):
            to_id = int(from_id + prob * class_size)
            if participant_id == pnumber - 1:
                lst[participant_id] += dct[class_id][from_id:to_id]  # to_id 
            else:
                lst[participant_id] += dct[class_id][from_id:to_id]
            from_id = to_id 
    
    siamese_datasets = [SiameseNetworkDataset(lst[i], dataset) for i in range(pnumber)]
    t_loaders = [torch.utils.data.DataLoader(siamese_datasets[i], batch_size=batch_size, shuffle=True) for i in range(pnumber)]

    # print("[data_loader.py: ] Number of common data points:", len(list(set(lst[0]) & set(lst[1])))) 
    
    # subsets = [torch.utils.data.Subset(dataset, lst[i]) for i in range(pnumber)]
    # t_loaders = [torch.utils.data.DataLoader(subsets[i], batch_size=batch_size, shuffle=True) for i in range(pnumber)]

    # print('Original data')
    # for pi in range(pnumber):
    #     counts = [0] * 10
    #     for (_, _, _, label) in siamese_datasets[pi]:
    #         counts[int(label[0].item())] += 1
    #     print(f'{pi+1} set: ', counts, sum(counts), end="\n")

    # print('Siamese pairs')
    # for pi in range(pnumber):
    #     counts = [0] * 10
    #     for (_, _, _, label) in siamese_datasets[pi]:
    #         counts[int(label[1].item())] += 1
    #     print(f'{pi+1} set: ', counts, sum(counts), end="\n")

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory,
    )

    return t_loaders + [train_loader]



def get_cifar10_test_loader(data_dir, batch_size, random_seed, num_workers=4, pin_memory=True, model_num=5, split='homogeneous'): 
    # define transforms
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4915, 0.4823, 0.4468], [0.2470, 0.2435, 0.2616])
    ])

    # load dataset
    dataset = datasets.CIFAR10(
        data_dir, train=False, download=True, transform=trans
    )
    
    pnumber = model_num 

    lst = []
    class_size = 200 
    num_classes = len(set(dataset.targets)) 

    # dictionary of labels map
    labels = dataset.targets
    dct = {}
    for idx, label in enumerate(labels):
        if label not in dct:
            dct[label] = []
        dct[label].append(idx)

    for i in range(num_classes):
        temp = random.sample(dct[i], len(dct[i]))
        dct[i] = temp

    # probabilities
    torch.set_printoptions(precision=3)
    probs = []
    for i in range(num_classes):
        probs.append([1.0 / pnumber] * pnumber)

    print(probs, end="\n\n")

    # division
    lst = {i: [] for i in range(pnumber)}
    for class_id, distribution in enumerate(probs):
        from_id = 0
        for participant_id, prob in enumerate(distribution):
            to_id = int(from_id + prob * class_size)
            if participant_id == pnumber - 1:
                lst[participant_id] += dct[class_id][from_id:to_id]  # to_id
            else:
                lst[participant_id] += dct[class_id][from_id:to_id]
            from_id = to_id

    subsets = [torch.utils.data.Subset(dataset, lst[i]) for i in range(pnumber)]
    t_loaders = [torch.utils.data.DataLoader(subsets[i], batch_size=batch_size, shuffle=False) for i in range(pnumber)]
    
    for pi in range(pnumber):
        counts = [0] * 10
        for label in subsets[pi]:
            counts[label[1]] += 1
        print(f'{pi+1} set: ', counts, sum(counts), end="\n")

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return t_loaders + [data_loader]