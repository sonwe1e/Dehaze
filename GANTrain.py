from tqdm import tqdm
import cv2
import torch
import argparse
from torch.utils.data import DataLoader
from AttUnetPP import UNet_Nested
from Data import DehazyDataset
from utils import PSNR, PSNRLoss, TVLoss, PerceptualLoss
from MSSSIM import MSSSIM
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2

torch.set_float32_matmul_precision('high')
parser = argparse.ArgumentParser(description='BuildingSeg')
parser.add_argument('--exp_name', type=str, default='test', metavar='N',
                    help='Name of the experiment')
parser.add_argument('-b', '--batch_size', type=int, default=10, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('-e', '--epochs', type=int, default=8000, metavar='N',
                    help='number of episode to train')
parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
                    help='learning rate')
parser.add_argument('--seed', type=int, default=413, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('-g', '--gpu_ids', type=str, default='cuda:0',
                    help='induct fix id to train')
parser.add_argument('--test', type=bool, default=False,
                    help='decide whether to test')
parser.add_argument('--lr_find', type=bool, default='',
                    help='decide whether to find lr')
args = parser.parse_args()

def main():
    
    # 定义参数记录
    run = wandb.init(
    project="Dehazing",
    name=args.exp_name,
    notes="",
    tags=["baseline"],
    config=args
)
        
    # 定义网络
    model = UNet_Nested().cuda(args.gpu_ids)
    model.load_state_dict(torch.load('Ckpt/test-1360.pth'))
    print(model)
    
    # 定义损失函数
    loss1 = PSNRLoss()
    loss2 = PerceptualLoss()
    loss3 = TVLoss()
    loss4 = MSSSIM()
    psnr_cal = PSNR()
    
    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    train_transform = A.Compose([
        A.RandomCrop(400, 400),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        ToTensorV2(transpose_mask=True)
    ])

    test_transform = A.Compose([
        A.RandomCrop(400, 400),
        ToTensorV2(transpose_mask=True)
    ])

    train_loader = DataLoader(DehazyDataset(mode='train', transform=train_transform), num_workers=40, pin_memory=True,
                            batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(DehazyDataset(mode='val', transform=test_transform), num_workers=40, pin_memory=True,
                            batch_size=1, shuffle=False)
    
    log_psnr = 0
    for epoch in tqdm(range(1, args.epochs+1)):
        train_loss = 0
        train_psnr = 0
        valid_loss = 0
        valid_psnr = 0
        loop = tqdm(enumerate(train_loader), total =len(train_loader))
        for _, (x, y) in loop:
            x, y = x.cuda(args.gpu_ids), y.cuda(args.gpu_ids)
            optimizer.zero_grad()
            out = model(x)
            loss = loss1(out, y) + loss2(out, y, x) + loss3(out) - loss4(out, y)
            psnr = psnr_cal(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / len(train_loader)
            train_psnr += psnr.item() / len(train_loader)
            loop.set_description(f'Epoch [{epoch}/{args.epochs}]')
            loop.set_postfix(train_loss = train_loss, train_psnr = train_psnr)
        wandb.log({"train_loss": train_loss, "train_psnr": train_psnr})
            
        loop = tqdm(enumerate(val_loader), total =len(val_loader))
        for _, (x, y) in loop:
            x, y = x.cuda(args.gpu_ids), y.cuda(args.gpu_ids)
            optimizer.zero_grad()
            out = model(x)
            loss = loss1(out, y) + loss2(out, y, x) + loss3(out) - loss4(out, y)
            psnr = psnr_cal(out, y)
            valid_loss += loss.item() / len(val_loader)
            valid_psnr += psnr.item() / len(val_loader)
            temp_img = out[0].permute(1, 2, 0).cpu().detach().numpy()
            cv2.imwrite(f'./test/{_}.png', 255*temp_img)
            loop.set_description(f'Epoch [{epoch}/{args.epochs}]')
            loop.set_postfix(valid_loss = valid_loss, valid_psnr = valid_psnr)
        wandb.log({"valid_loss": valid_loss, "valid_psnr": valid_psnr})
        if log_psnr < valid_psnr:
            log_psnr = valid_psnr
            wandb.log({"best_val_psnr": valid_psnr})
            torch.save(model.state_dict(), f'./Ckpt/{args.exp_name}-{epoch:04d}.pth')
        wandb.log({"epoch": epoch})
        # print(f'epoch: {epoch}, train_loss: {loss.item()}, train_psnr: {psnr.item()}')
        # print(f'epoch: {epoch}, val_loss: {loss.item()}, val_psnr: {psnr.item()}')
            

if __name__ == '__main__':
    main()