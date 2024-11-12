import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
from sklearn.metrics import confusion_matrix
from collections import defaultdict


from config import get_config 
from models import resnet34_model
from data_loader import get_cifar10_train_loader, get_cifar10_test_loader 
import copy

from utils import ContrastiveLoss, plot_pca_embeddings
import wandb 

from tqdm import tqdm
import random 

import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

# initialization 
config, unparsed = get_config() 

num_workers = 4 
pin_memory = True 
model_num = config.model_num 
split = config.split 
batch_size = config.batch_size 
random_seed = config.random_seed 
use_tensorboard = config.use_tensorboard 
use_wandb = config.use_wandb 
aggregation = config.aggregation

data_dir = 'data/cifar10/'
file_name = config.save_name #f'cifar10_p{model_num}_batch{batch_size}_{split}_seed{random_seed}_agg{aggregation}' 

# Check if CUDA is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(random_seed)

# if using wandb 
if use_wandb:
    wandb.init(project="embedding", name=f"{file_name}") 

# data loader 
kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory, 'model_num': model_num, 'split': split} 
test_data_loader = get_cifar10_test_loader(data_dir, batch_size, random_seed, **kwargs)
train_data_loader = get_cifar10_train_loader(data_dir, batch_size, random_seed, shuffle=True, **kwargs) 

def classwise_accuracy(model, dataloader, client_prototypes):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.cuda(), targets.cuda()
            embeddings, _ = model(data, data)

            for emb, true_class in zip(embeddings, targets):
                # Calculate distances to each class mean embedding
                distances = {cls: torch.norm(emb.cpu() - mean_embedding) 
                             for cls, mean_embedding in client_prototypes.items()}
                
                # Predict the nearest class mean
                predicted_class = min(distances, key=distances.get)
                
                all_preds.append(predicted_class)
                all_targets.append(true_class.item())

    cm = confusion_matrix(all_targets, all_preds)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    return class_acc

class Client:
    def __init__(self, dataloader, client_id):
        self.model = resnet34_model().cuda()
        self.dataloader = dataloader
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01) 
        self.client_id = client_id
        self.old_weights = None
        self.old_prototypes = None

    def train(self, round, server_prototypes, epochs=1): 
        self.old_weights = copy.deepcopy(self.model.state_dict())
        self.model.train()
        criterion = ContrastiveLoss()
        mse_criterion = nn.MSELoss()
        embeddings = defaultdict(list)

        lambda_reg = config.lambda_reg  # Regularization parameter, adjust as needed

        if (round > 1) and (config.flag != 3):
            prototype_tensor = torch.stack([server_prototypes[i] for i in range(10)]).cuda()

        for epoch in range(epochs):
            total_loss_value = 0.0  
            total_loss1 = 0.0 # contrastive
            total_loss2 = 0.0 # mse
            total_loss3 = 0.0 # mse
            num_batches = 0  

            for img0, img1, flag, targets in self.dataloader:
                img0, img1, flag, target1, target2 = img0.cuda(), img1.cuda(), flag.cuda(), targets[:, 0].cuda(), targets[:, 1].cuda()
                
                self.optimizer.zero_grad()
                output1, output2 = self.model(img0, img1)
                
                loss1 = criterion(output1, output2, flag)

                if (round > 1) and (config.flag != 3):
                    server_proto1 = prototype_tensor[target1.long()]
                    server_proto2 = prototype_tensor[target2.long()]

                    loss2 = lambda_reg * mse_criterion(output1, server_proto1)
                    loss3 = lambda_reg * mse_criterion(output2, server_proto2)

                    total_loss = loss1 + loss2 + loss3
                    total_loss2 += loss2.item()
                    total_loss3 += loss3.item()
                else:
                    total_loss = loss1
                
                total_loss.backward()
                self.optimizer.step()

                # Accumulate losses
                total_loss_value += total_loss.item()
                total_loss1 += loss1.item()
                
                num_batches += 1  # Increment batch counter
                
                if epoch == epochs - 1:
                    for emb, cls in zip(output1, target1):
                        embeddings[cls.item()].append(emb.detach().cpu())
                    for emb, cls in zip(output2, target2):
                        embeddings[cls.item()].append(emb.detach().cpu())

            # Calculate average losses for the epoch
            average_total_loss = total_loss_value / num_batches if num_batches > 0 else 0.0
            average_loss1 = total_loss1 / num_batches if num_batches > 0 else 0.0
            average_loss2 = total_loss2 / num_batches if num_batches > 0 else 0.0
            average_loss3 = total_loss3 / num_batches if num_batches > 0 else 0.0
            
            if use_wandb:
                # Log losses to WandB
                wandb.log({
                    f"train_loss/total_loss_{self.client_id}": average_total_loss,
                    f"train_loss/loss_1_{self.client_id}": average_loss1,
                    f"train_loss/loss_2_{self.client_id}": average_loss2,
                    f"train_loss/loss_3_{self.client_id}": average_loss3,
                    "round": round,
                    "epoch": epoch
                })

        client_prototypes = {}
        final_embeddings = {} # embeddings of all datapoints in all clients
        for cls, emb_list in embeddings.items():
            final_embeddings[cls] = torch.stack(emb_list).cpu()
            client_prototypes[cls] = torch.mean(torch.stack(emb_list), dim=0).cpu()

        # if self.old_prototypes is None:
        #     self.old_prototypes = client_prototypes
        # else:
        #     for cls  in self.old_prototypes:
        #         self.old_prototypes[cls] = 0.9 * self.old_prototypes[cls] + 0.1 * client_prototypes[cls]

        return client_prototypes, final_embeddings

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)
    
    def get_gradients(self): 
        # Collecting and returning the gradients
        new_weights = self.get_weights()
        old_weights = self.old_weights
        gradients = {old_name: torch.div(old_param - new_param, 0.01) for ((old_name, old_param), (new_name, new_param)) in zip(old_weights.items(), new_weights.items())}
        return gradients
        gradients = {name: param.grad.clone() for name, param in self.model.named_parameters() if param.grad is not None}
        return gradients
    

# Server Class
class Server:
    def __init__(self, clients):
        self.model = resnet34_model().cuda()
        self.clients = clients

        self.stored_grads = None 

    def aggregate_gradients(self, coefficients):
        total_grads = None

        for client_id, client in enumerate(self.clients): 
            client_grads = client.get_gradients() 

            if total_grads is None:
                total_grads = {name: torch.zeros_like(grad) for name, grad in client_grads.items()}

            for name, grad in client_grads.items():
                total_grads[name] += coefficients[client_id] * grad

        # Store the aggregated gradients
        self.stored_grads = total_grads

    def aggregate(self, coefficients):
        total_weights = None

        for client_id, client in enumerate(self.clients): 
            client_weights = client.get_weights()

            if total_weights is None:
                total_weights = {name: torch.zeros_like(param) for name, param in client_weights.items()}

            for name, param in client_weights.items():
                if  total_weights[name].dtype == torch.float32:
                    total_weights[name] += coefficients[client_id] * param.to(torch.float32)
                elif total_weights[name].dtype == torch.int64:
                    total_weights[name] += int(coefficients[client_id]) * param.to(torch.int64)

        prev_weights = self.model.state_dict()
        eta = 1.0 
        for name, param in total_weights.items():
            prev_weights[name] = (1 - eta) * prev_weights[name] + eta * param 

        self.model.load_state_dict(prev_weights) 

    def broadcast(self, coefficients=None): 
        for client in self.clients: 
            for (global_param, client_param) in zip(self.model.parameters(), client.model.parameters()):
                # personalization 
                client_param.data = global_param.data #(1 - coefficients[client.client_id]) * client_param.data + coefficients[client.client_id] * global_param.data 

    
    def evaluate(self, dataloader, server_prototypes):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.cuda(), targets.cuda()
                embeddings, _ = self.model(data, data)

                for emb, true_class in zip(embeddings, targets):
                    # Calculate distances to each class mean embedding
                    distances = {cls: torch.norm(emb.cpu() - mean_embedding) 
                                for cls, mean_embedding in server_prototypes.items()}
                    
                    # Predict the nearest class mean
                    predicted_class = min(distances, key=distances.get)
                    
                    all_preds.append(predicted_class)
                    all_targets.append(true_class.item())

        cm = confusion_matrix(all_targets, all_preds)
        class_acc = cm.diagonal() / cm.sum(axis=1)
        avg_acc = np.mean(class_acc) * 100
        print(class_acc)
        return avg_acc, 0.0, class_acc


client_loaders = train_data_loader 

clients = [Client(loader, i) for i, loader in enumerate(client_loaders[:-1])][:model_num]
server = Server(clients) 
for client in clients:
    client.set_weights(server.model.state_dict()) 
    
weights = [1 / model_num] * model_num 
shapley_values, mu = None, 0.5 
freq_rounds = None 

num_rounds = config.num_rounds 
num_lepochs = [config.num_lepochs] * model_num 

server_prototypes = defaultdict(list)

for round in range(num_rounds):
    client_prototypes = [] # list of dictionary for each client
    all_embeddings = []
    if round > 0:
        num_lepochs = [1] * model_num # commented in prev experiments

    for cl_idx in tqdm(range(len(clients)), desc=f"Training Clients in Round {round + 1}"):
        client = clients[cl_idx]
        c_prototypes, final_embeddings = client.train(round=round, epochs=num_lepochs[cl_idx], server_prototypes=server_prototypes)
        client_prototypes.append(c_prototypes)
        all_embeddings.append(final_embeddings)
        
    print('#' * 100)
    server_prototypes = defaultdict(list)
    # evaluating clients 
    for idx, client in enumerate(clients):
        client_val_accuracy = classwise_accuracy(client.model, test_data_loader[-1], client_prototypes[idx])  # Use the client's model for evaluation
        print(f"[Client {idx}] Round {round + 1}/{num_rounds}, Balanced Accuracy: {np.mean(client_val_accuracy)*100:.2f}%, {client_val_accuracy}")

        # wandb 
        if use_wandb:
            wandb.log({f"balanced_valid_acc_{idx}": np.mean(client_val_accuracy) * 100, "round": round}) 
            for j, acc in enumerate(client_val_accuracy):
                wandb.log({f"client_{idx}/class_acc_{j}": acc, "round": round}) 
        
    
    for client_embeddings in client_prototypes:
        for cls, emb in client_embeddings.items():
            server_prototypes[cls].append(emb)

    for cls, emb_list in server_prototypes.items():
        server_prototypes[cls] = torch.mean(torch.stack(emb_list), dim=0)

    # continue

    # client_prototypes.append(server_prototypes) # 2 clients : 20 prototypes, 10 server aggregated prototype = (30, 128)
    all_embeddings.append(server_prototypes) # 2 clients : all data + 10 server aggregated prototypes
    all_embeddings.extend(client_prototypes)


    # plot_pca_embeddings(client_prototypes, round, save_dir=f"{config.lambda_reg}_pca_plots_prototypes_tmp")
    plot_pca_embeddings(all_embeddings, round, model_num, save_dir=config.save_name)
    server.aggregate(coefficients = weights)


    # here, we just used server.evaluate, since the test set is balanced; 
    # val_accuracy, val_loss, _ = server.evaluate(test_data_loader[-1], server_prototypes) 
    # print(f"Round {round + 1}/{num_rounds}, Validation Accuracy: {val_accuracy:.2f}%, Validation Loss: {val_loss:.4f}") 
    
    server_val_accuracy = classwise_accuracy(server.model, test_data_loader[-1], client_prototypes[-1])  # Use the client's model for evaluation
    print(f"[Server] Round {round + 1}/{num_rounds}, Balanced Accuracy: {np.mean(server_val_accuracy)*100:.2f}%, {server_val_accuracy}")
    val_accuracy = np.mean(server_val_accuracy)*100

    # wandb 
    if use_wandb:
        wandb.log({f"valid_acc": val_accuracy, "round": round}) 
        # wandb.log({f"valid_loss": val_loss, "round": round}) 

    if config.flag != 1:
        server.broadcast()