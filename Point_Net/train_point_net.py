from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from data import PcDataset
from pnet2 import PointNetCls
from torch.utils.data.dataloader import DataLoader
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import numpy as np
import os
import sys

class Trainer():
    def __init__(self, net, n_epochs, optimizer, 
                lr_scheduler, scheduler_step, loss_fn, writer):
        self.net = net
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scheduler_step = scheduler_step
        self.loss_fn = loss_fn
        self.writer = writer
        self.global_step = 0
    
    def train_step(self, train_loader):
        step_loss = 0.0
        for step, (data, labels) in enumerate(train_loader):
            self.optimizer.zero_grad()
            data, labels = data.cuda(), labels.cuda()
            estimates, _, _  = self.net(data)
            # reshape estimates
            loss =  self.loss_fn(estimates.float(), labels.float())
            #print("train_loss",loss.item(), type(loss))
            loss.backward()
            self.optimizer.step()
            step_loss += loss.item()
            self.global_step += 1
        avg_loss = step_loss/len(train_loader)
        print('train_loss', avg_loss)
        
        self.writer.add_scalar('train_loss', avg_loss, step)
        self.writer.close()
        return step_loss,

    def val_step(self, val_loader):
        step_loss = 0.0
        for step, (data, labels) in enumerate(val_loader):
            data, labels = data.cuda(), labels.cuda()
            estimates, _, _  = self.net(data)
            loss = self.loss_fn(estimates.float(), labels.float())
            #print("val_steploss",loss.item())
            step_loss += loss.item()
        avg_loss = step_loss/len(val_loader)
        print('val_loss', avg_loss)
        self.writer.add_scalar('val_loss', loss.item(), step)
        self.writer.close()
        return step_loss

    def train(self, train_loader, val_loader):
        for i in range(self.n_epochs):
            # train step
            self.train_step(train_loader)
            # LR update 
            ## update lr after seeing 1000 extra examples
            if (self.global_step + 1) % self.scheduler_step == 0:
                self.scheduler.step()
                print("Learning rate", self.scheduler.get_lr())
             # eval step 
            with torch.no_grad():
                self.val_step(val_loader)

class TestPointNet():
    def __init__(self, pc_file, num_classes, ckpt_file, 
                landmark_file, predict=False):
        self.pc_file = pc_file
        self.num_classes = num_classes
        self.lnd_names_file = landmark_file
        self.dataset = PcDataset(self.pc_file, size=256000)
        self.ckpt_file = ckpt_file
        data = np.load(pc_file, allow_pickle=True)
        if predict:
            self.pc_data = data
        else:
            self.pc_data, self.tok_ids, self.landmarks = data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # get all landmark names
        f = open(self.lnd_names_file, 'r')
        self.names = sorted([x.strip() for x in f.readlines()])
    
    def predict_landmarks(self):
        self.num_3d_points = self.pc_data.shape[0]
        self.net = PointNetCls(self.num_classes, 
                            feature_transform=True, pc_size=self.num_3d_points)
        self.net.load_state_dict(
            torch.load(self.ckpt_file, map_location=self.device).module.state_dict())
        self.net.eval()
        out, _, _ = net(self.pc_data.unsqueeze(0))
        self.tok_ids = [x-1 for x in self.tok_ids] # tok_ids is 1 indexed not zero-indexed
        
        vals = {idx: torch.linalg.norm(out.squeeze(0)[idx] - \
                 torch.tensor(self.landmarks)[idx]).item() for idx in self.tok_ids}
        errors = {self.names[idx]: vals[idx] for idx in self.names}
        return errors

if __name__=='__main__':
    n_epochs = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    scheduler_step = int(sys.argv[3])
    num_points_per_pc = int(sys.argv[4])
    train_file = sys.argv[5]
    val_file = sys.argv[6]
    log_dir = sys.argv[7]
    # model config
    #config = {'conv': [64, 128, 1024, 256, 512, 1024, 64, 128, 128, 512, 1048], 
    #        'fc': [128, 128, 256, 512, 256]}
    num_classes = 128*3
    # Data Loaders 
    train_set = PcDataset(train_file,num_points_per_pc)
    val_set = PcDataset(val_file,num_points_per_pc)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                        shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                        shuffle=False, num_workers=4, drop_last=True)
    # Training Ops
    # net = PointNet(num_classes, config) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} . . .")
    net = nn.DataParallel(PointNetCls(num_classes, feature_transform=True))
    net.to(device)
    print(net)  
    
    optimizer = Adam(net.parameters(), lr=1e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.7, verbose=False)
    loss_fn = nn.MSELoss()
    writer = SummaryWriter(logdir=log_dir)

    trainer = Trainer(net, n_epochs, optimizer, scheduler, scheduler_step, loss_fn, writer)
    trainer.train(train_loader, val_loader)
    # save model
    model_path = os.path.join(os.path.abspath(""), "model2.pt")
    torch.save(net, model_path)



"""
1st holistic attempt:
train_loss 0.016809802555995023
val_loss 0.005139501238290926
"""