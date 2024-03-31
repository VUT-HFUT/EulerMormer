import torch.optim
import torch.utils.data as data
from utils.data_loader import ImageFromFolder
from utils.avgMeter import AverageMeter
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from config import Config
from models.magnet import MagNet
from callbacks import save_model, gen_state_dict
from tensorboardX import SummaryWriter
writer = SummaryWriter(log_dir="./log_magnet_Euler")
from utils.utils import EdgeLoss, CharbonnierLoss

# Configurations
losses = []
def main():
    config = Config()
    cudnn.benchmark = True
    magnet = MagNet()
    start_epoch = 0
    if config.pretrained_weights:
        print("=> loading checkpoint '{}'".format(config.pretrained_weights))
        magnet.load_state_dict(gen_state_dict(config.pretrained_weights))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        magnet = nn.DataParallel(magnet)
    else:
        print("=> no checkpoint found at '{}'".format(config.pretrained_weights))

    magnet.to(device)
    print("using device{}".format(device))
    print(magnet) 

    # Metrics
    criterion_char = CharbonnierLoss().cuda()
    criterion_edge = EdgeLoss().cuda()
    optimizer = optim.Adam(magnet.parameters(), lr=config.lr,
                                 betas=config.betas,
                                 weight_decay=config.weight_decay)
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    print('Save_dir:', config.save_dir)

    # Data generator
    dataset_mag = ImageFromFolder(
        config.dir_train, num_data=config.numdata, preprocessing=True)
    data_loader = data.DataLoader(dataset_mag,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.workers,
                                  pin_memory=False)

    # Summary of the system =====================================================
    print('===================================================================')
    print('PyTorch Version: ', torch.__version__)
    #print('Torchvision Version: ',torchvision.__version__)
    print('===================================================================')

    # Summary of the model ======================================================
    print('Network parameters {}'.format(sum(p.numel()
          for p in magnet.parameters())))
    print('Trainable network parameters {}'.format(sum(p.numel()
          for p in magnet.parameters() if p.requires_grad)))

          
    losses, losses_recon, losses_AC, losses_BC, losses_Edge = [], [], [], [],[]
    # Training
    for epoch in range(start_epoch, config.epochs):
        loss, loss_recon, loss_AC, loss_BC, loss_Edge = train(
            data_loader, magnet, criterion_char,criterion_edge, optimizer, epoch, device, config)

        # Stack losses
        losses.append(loss)
        losses_recon.append(loss_recon)
        losses_AC.append(loss_AC)
        losses_BC.append(loss_BC)
        losses_Edge.append(loss_Edge)
        magnet.cuda()  # Return model to device
        save_model(magnet.state_dict(), losses, config.save_dir, epoch)


def train(loader, model, criterion_char,criterion_edge, optimizer, epoch, device, config):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_recon = AverageMeter()
    losses_AC = AverageMeter()  # B - C loss
    losses_BC = AverageMeter()  # B - C loss
    losses_Edge = AverageMeter()  # B - C loss

    model.train()

    end = time.time()
    for i, (y, xa, xb, xc, mag_factor) in enumerate(loader):
        y = y.cuda()
        xa = xa.to(device)
        xb = xb.to(device)
        xc = xc.to(device)
        mag_factor = mag_factor.to(device)
        data_time.update(time.time() - end)

        # Compute output
        mag_factor = mag_factor.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        y_hat, v_a, v_c, m_b, m_c= model(xa, xb, mag_factor, xc, mode='train')

        # Compute losses
        loss_recon = criterion_char(y_hat, y)
        loss_AC = 0.1 * criterion_char(v_a, v_c)
        loss_BC = 0.1 * criterion_char(m_b, m_c)
        loss_Edge = 0.5 * criterion_edge(y_hat, y)
        loss = loss_recon + loss_AC + loss_BC + loss_Edge
        losses.update(loss.item())
        losses_recon.update(loss_recon.item())
        losses_AC.update(loss_AC.item())
        losses_BC.update(loss_BC.item())
        losses_Edge.update(loss_Edge.item())

        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.num_val_per_epoch == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'LossAC {loss_AC.val:.4f} ({loss_AC.avg:.4f})\t'
                  'LossBC {loss_BC.val:.4f} ({loss_BC.avg:.4f})\t'
                  'LossEdge {loss_Edge.val:.4f} ({loss_Edge.avg:.4f})\t'.format(
                      epoch, i, len(loader), batch_time=batch_time, data_time=data_time,
                      loss=losses_recon, loss_AC=losses_AC, loss_BC=losses_BC,loss_Edge = losses_Edge))
            writer.add_scalar('train_loss_recon_epoch', losses_recon.avg, epoch)
            writer.add_scalar('train_losses_AC_epoch', losses_AC.avg, epoch)
            writer.add_scalar('train_losses_BC_epoch', losses_BC.avg, epoch)
            writer.add_scalar('train_losses_Edge_epoch', losses_Edge.avg, epoch)

    return losses.avg, losses_recon.avg, losses_AC.avg, losses_BC.avg,losses_Edge.avg


if __name__ == '__main__':
    main()