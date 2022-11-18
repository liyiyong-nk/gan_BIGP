import argparse, itertools, os, time
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage import img_as_ubyte

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import torch
import torch.nn as nn
import cv2
from models.models import Generator, Discriminator
from utils.utils import *
from utils.perceptual import *
from utils.fid_score import calculate_fid_given_paths
from datasets.datasets import ImageDataset, PairedImageDataset


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='7')
parser.add_argument('--cpus', default=4)
parser.add_argument('--batch_size', '-b', default=8, type=int, help='mini-batch size')
parser.add_argument('--epochs', '-e', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--gamma_round', default=1000, type=int, help='round of gamma lr consine')
parser.add_argument('--lrw', type=float, default=1e-5, help='learning rate for G')
parser.add_argument('--lrgamma', type=float, default=1e-1, help='learning rate for gamma (pruning)')
parser.add_argument('--wd', type=float, default=1e-3, help='weight decay for G')
parser.add_argument('--momentum', default=0.5, type=float, help='momentum')
parser.add_argument('--dataset', type=str, default='horse2zebra', choices=['summer2winter_yosemite', 'horse2zebra'])
parser.add_argument('--task', type=str, default='A2B', choices=['A2B', 'B2A'])
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--rho', type=float, default=0, help='l1 loss weight')
parser.add_argument('--beta', type=float, default=20, help='GAN loss weight')
parser.add_argument('--alpha', type=float, default=4, help='BIG loss weight')
parser.add_argument('--lc', default='vgg', choices=['vgg', 'mse'], help='G content loss. vgg: perceptual; mse: mse')
parser.add_argument('--quant', action='store_true', help='enable quantization (for both activation and weight)')
parser.add_argument('--resume', action='store_true', help='If true, resume from early stopped ckpt')
parser.add_argument('--name', default='tspv1', help='name to distinguish ckpt dir')
parser.add_argument('--contral_rate', type=float, default=0.001, help='gamma threshold = 0.5/255')
parser.add_argument('--test_image_dir', default='./datasets/horse2zebra/test/A/')
parser.add_argument('--image_gt_dir', default='./datasets/horse2zebra/test/B/')
# parser.add_argument('--test_image_dir', default='./datasets/summer2winter_yosemite/test/B/')
# parser.add_argument('--image_gt_dir', default='./datasets/summer2winter_yosemite/test/A/')


args = parser.parse_args()
if args.task == 'A2B':
    source_str, target_str = 'A', 'B'
else:
    source_str, target_str = 'B', 'A'
foreign_dir = './'
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

## results_dir:
method_str = ('GS8' if args.quant else 'GS32')
gamma_optimizer_str = 'sgd_mom%s_lrgamma%s' % (args.momentum, args.lrgamma)
W_optimizer_str = 'adam_lrw%s_wd%s' % (args.lrw, args.wd)
opt_str = 'e%d-b%d' % (args.epochs, args.batch_size)
loss_str = 'rho%s_beta%s_%s' % (args.rho, args.beta, args.lc)
alpha_str = 'alpha%s' % (args.alpha)
contral_rate_str = 'contral_rate%s' % (args.contral_rate)
# results_dir = os.path.join('results', args.dataset, args.task, '%s_%s_%s_%s_%s_%s_%s_%s' % (
#     method_str, loss_str, opt_str, gamma_optimizer_str, W_optimizer_str, alpha_str, contral_rate_str, args.name))
results_dir = os.path.join('results', args.dataset, args.task, '%s_%s_%s' % (
    alpha_str, contral_rate_str, args.name))
img_dir = os.path.join(results_dir, 'img')
pth_dir = os.path.join(results_dir, 'pth')
create_dir(img_dir), create_dir(pth_dir)
file_handle1 =open(results_dir+'/paramater.lst', mode='w+', encoding='utf8')
## Networks
# G:
netG = Generator(args.input_nc, args.output_nc, quant=args.quant).cuda()
# D:
netD = Discriminator(args.input_nc).cuda()

# param list:
parameters_G, parameters_D, parameters_gamma = [], [], []
for name, para in netG.named_parameters():
    if 'weight' in name and para.ndimension() == 1:
        parameters_gamma.append(para)
    else:
        parameters_G.append(para)
for name, para in netD.named_parameters():
    # print(name, para.size(), para.ndimension())
    parameters_D.append(para)
print('parameters_gamma:', len(parameters_gamma))

# Optimizers:
optimizer_gamma = torch.optim.SGD(parameters_gamma, lr=args.lrgamma, momentum=args.momentum)
optimizer_G = torch.optim.Adam(parameters_G, lr=args.lrw, weight_decay=args.wd, betas=(0.5, 0.999)) # lr=1e-3
optimizer_D = torch.optim.Adam(parameters_D, lr=args.lrw, weight_decay=args.wd, betas=(0.5, 0.999)) # lr=1e-3

# LR schedulers:
lr_scheduler_gamma = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gamma, args.gamma_round)
lr_scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, args.epochs)
lr_scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, args.epochs)

# load pretrained:
if args.resume:
    last_epoch, loss_G_lst, loss_G_perceptual_lst, loss_G_GAN_lst, loss_D_lst, channel_number_lst = load_ckpt(
        netG, netD,
        optimizer_G, optimizer_D, optimizer_gamma,
        lr_scheduler_G, lr_scheduler_D, lr_scheduler_gamma,
        path=os.path.join(results_dir, 'pth', 'latest.pth')
    )
    start_epoch = last_epoch + 1
else:
    if args.dataset == 'horse2zebra':
        # dense_model_folder = 'pretrained_dense_model_quant' if args.quant else 'pretrained_dense_model'
        g_path = '/data1/liyiyong/gan_BIGP/pretrained_dense_model/horse2zebra/netG_A2B_epoch_199.pth'
        netG.load_state_dict(torch.load(g_path))
        print('load G from %s' % g_path)
        d_path = '/data1/liyiyong/gan_BIGP/pretrained_dense_model/horse2zebra/netD_B_epoch_199.pth'
        netD.load_state_dict(torch.load(d_path))
        print('load D from %s' % d_path)
    elif args.dataset == 'summer2winter_yosemite':
        g_path = '/data1/liyiyong/gan_BIGP/pretrained_dense_model/summer2winter_yosemite/netG_A2B_epoch_199.pth'
        netG.load_state_dict(torch.load(g_path))
        print('load G from %s' % g_path)
        d_path = '/data1/liyiyong/gan_BIGP/pretrained_dense_model/summer2winter_yosemite/netD_B_epoch_199.pth'
        netD.load_state_dict(torch.load(d_path))
        print('load D from %s' % d_path)
    start_epoch = 0
    loss_G_lst, loss_G_perceptual_lst, loss_G_GAN_lst, loss_D_lst, channel_number_lst = [], [], [], [], []
FID_gt_lst = []
# Dataset loader: image shape=(256,256)
dataset_dir = os.path.join(foreign_dir, 'datasets', args.dataset)
soft_data_dir = os.path.join(foreign_dir, 'train_set_result', args.dataset)
transforms_ = [ transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ] # (0,1) -> (-1,1)
dataloader = DataLoader(
    PairedImageDataset(dataset_dir, soft_data_dir, transforms_=transforms_, mode=args.task),
    batch_size=args.batch_size, shuffle=True, num_workers=args.cpus, drop_last=True)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor
input_source = Tensor(args.batch_size, args.input_nc, args.size, args.size)
input_target = Tensor(args.batch_size, args.output_nc, args.size, args.size)
target_real = Variable(Tensor(args.batch_size).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(args.batch_size).fill_(0.0), requires_grad=False)
fake_img_buffer = ReplayBuffer()

# perceptual loss models:
vgg = VGGFeature().cuda()

# soft threshold of in
def update_in(model, sparse_loss_dict, sparse_loss_dict_index):
    idx = 0
    height = args.size
    width = args.size
    for m in model.modules():
        if isinstance(m, nn.InstanceNorm2d):
            idx += 1
            if m.weight is not None and idx in sparse_loss_dict:
                gamma1 = sparse_loss_dict[idx]
                gamma_max = 0.0
                beta_max = 0.0
                gamma_bord = 0.0
                beta_bord = 0.0
                gamma1_sorted, indices = torch.sort(gamma1)
                tensor_size = len(gamma1)
                for i in range(tensor_size):
                    if sparse_loss_dict_index[idx][i]==1:
                        beta_bord = beta_bord + gamma1[i]
                        beta_max = max(beta_max,gamma1[i])
                    else:
                        gamma_bord = gamma_bord + gamma1[i]
                        gamma_max = max(gamma_max,gamma1[i])

                gamma_bord = gamma_bord*args.contral_rate
                beta_bord = beta_bord*args.contral_rate/10
                gamma_max = gamma_max/10
                beta_max = beta_max/20
                # gamma_max = gamma_max/8
                # beta_max = beta_max/16
                # gamma_max = gamma_max/6
                # beta_max = beta_max/15
                cnt = 0
                for i in range(tensor_size):
                    tem_sum = gamma1_sorted[:i].sum()
                    if tem_sum > beta_bord and tem_sum > gamma_bord:
                        break
                    if sparse_loss_dict_index[idx][indices[i]]==1:
                        if tem_sum <= beta_bord and gamma1[indices[i]] < beta_max:
                            m.weight.data[indices[i]] = 0.0
                            m.bias.data[indices[i]] = 0.0
                            cnt += 1
                    else:
                        if tem_sum <= gamma_bord and gamma1[indices[i]] <= gamma_max:
                            m.weight.data[indices[i]] = 0.0
                            m.bias.data[indices[i]] = 0.0
                            cnt += 1
                
        if isinstance(m, nn.Conv2d):
            width /= m.stride[0]
            height /= m.stride[1]    
            # print ('layer = {}, zero cnt = {}'.format(idx, cnt))
    # print ('=' * 100)

# paper's BIG loss
def BIG_loss(model):
    loss_nowg = 0.0
    loss_nowb = 0.0
    loss_nowwg = 0.0
    loss_nowwb = 0.0
    idx = 0
    loss = 0.0
    now_pp = 0
    now_qq = 0
    now_con = 0
    p = 0
    q = 0
    height = args.size
    width = args.size
    sparse_loss_dict = {}
    sparse_loss_dict_index = {}
    channel = 0
    for m in model.modules():
        if isinstance(m, nn.InstanceNorm2d):
            p_in = m.weight
            q_in = m.bias
            idx += 1
            if p_in is None:
                # print ('None')
                now_pp = 0.0
            else:
                now_pp = p_in.abs()
                # loss_nowg = loss_nowg+now_pp.sum()
            if q_in is None:
                # print ('None')
                channel = 0
                now_qq = 0.0
            else:
                now_qq = q_in.abs()
                channel = len(q_in) 
                # loss_nowb = loss_nowb+now_qq.sum()

        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            p_conv = m.weight

            now_con1 = p_conv * p_conv
            now_con = p_conv.abs()
            # now_con = p_conv
            if isinstance(m, nn.Conv2d):
                now_conb = now_con[:, :, :,:].sum(axis=(2,3))
                now_conb1 = now_conb.abs()
                now_conb2 = now_conb1[:, :].sum(axis=(0))
                now_cong = now_con1[:, :, :,:].sum(axis=(2,3))+0.0000001
                now_cong1 = now_cong.sqrt()
                now_cong2 = now_cong1[:, :].sum(axis=(0))
            else:
                now_conb = now_con[:, :, :,:].sum(axis=(2,3))
                now_conb1 = now_conb.abs()
                now_conb2 = now_conb1[:, :].sum(axis=(1))
                now_cong = now_con1[:, :, :,:].sum(axis=(2,3))+0.0000001
                now_cong1 = now_cong.sqrt()
                now_cong2 = now_cong1[:, :].sum(axis=(1))
                
            t = np.sqrt(width*height)

            # loss_nowwg = loss_nowwg+now_cong2.sum()
            # loss_nowwb = loss_nowwb+now_conb2.sum()
            loss_now = now_pp* now_cong2 + now_qq * now_conb2
            # loss_now = (now_pp + now_qq) * now_con
            sparse_loss_dict[idx] = loss_now
            sparse_loss_dict_index[idx] = []
            for j in range(channel):
                if q_in[j] <= -now_pp[j] * t:
                    sparse_loss_dict[idx][j] = 0
                    sparse_loss_dict_index[idx].append(0)
                elif q_in[j] > now_pp[j] * t and now_pp[j]>0:
                    sparse_loss_dict[idx][j] = now_pp[j]* now_cong2[j]
                    sparse_loss_dict_index[idx].append(1)
                elif now_pp[j]==0:
                    sparse_loss_dict[idx][j] = 0
                    sparse_loss_dict_index[idx].append(0)
                else:
                    sparse_loss_dict_index[idx].append(2)
            loss = loss + sparse_loss_dict[idx].sum()
            
            now_pp = 0.0
            now_qq = 0.0
            width /= m.stride[0]
            height /= m.stride[1]
    # print(loss,loss_nowg,loss_nowb,loss_nowwg,loss_nowwb)
    return loss, sparse_loss_dict, sparse_loss_dict_index

def adjust_dynamic_range (data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
            np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data


def to_range(images, min_value=0.0, max_value=1.0, dtype=None):
    assert \
        np.min(images) >= -1.0 - 1e-5 and np.max(images) <= 1.0 + 1e-5 \
        and (images.dtype == np.float32 or images.dtype == np.float64), \
        'The input images should be float64(32) and in the range of [-1.0, 1.0]!'
    if dtype is None:
        dtype = images.dtype
    return ((images + 1.) / 2. * (max_value - min_value) + min_value).astype(dtype)


def get_image_file(folder):
    imagelist =[]
    for parent, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            if filename.lower().endswith(('.jpg', '.jpeg','.png', '.webp')):
                imagelist.append(os.path.join(parent, filename))
    return imagelist

def inference(image_path, model):
    # img = Image.open(image_path)
    # img = img.convert("RGB")
    img_np = cv2.imread(image_path)
    img_np = adjust_dynamic_range(img_np.astype(np.float32), [0, 255], [-1., 1.])
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_np = img_np[np.newaxis, :, :, :]
    img_np = img_np.transpose(0, 3, 1, 2)
    img = torch.from_numpy(img_np.copy()).cuda()
    with torch.no_grad():
        res = model(img)

    res_numpy = res.cpu().float().detach().numpy()
    res_numpy = res_numpy.transpose(0, 2, 3, 1)
    res_numpy = np.clip(res_numpy, -1, 1)
    out = to_range(res_numpy, 0, 255, np.uint8)[0,:,:,:]
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out

###### Training ######
print('dataloader:', len(dataloader)) # 1334
for epoch in range(start_epoch, args.epochs):
    start_time = time.time()
    netG.train(), netD.train()
    # define average meters:
    loss_G_meter, loss_G_perceptual_meter, loss_G_GAN_meter, loss_D_meter = \
        AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    for i, batch in enumerate(dataloader):
        # Set model input
        input_img = Variable(input_source.copy_(batch[source_str])) # X
        teacher_output_img = Variable(input_target.copy_(batch[target_str])) # Gt(X)

        # print('input_img:', input_img.size(), torch.max(input_img), torch.min(input_img))
        # print('teacher_output_img:', teacher_output_img.size(), torch.max(teacher_output_img), torch.min(teacher_output_img))

        ###### G ######
        optimizer_G.zero_grad()
        optimizer_gamma.zero_grad()

        student_output_img = netG(input_img) # Gs(X)

        # perceptual loss
        if args.lc == 'vgg':
            student_output_vgg_features = vgg(student_output_img)
            teacher_output_vgg_features = vgg(teacher_output_img)

            loss_G_perceptual, loss_G_content, loss_G_style = \
                perceptual_loss(student_output_vgg_features, teacher_output_vgg_features)
        elif args.lc == 'mse':
            loss_G_perceptual = F.mse_loss(student_output_img, teacher_output_img)

        # GAN loss (G part):
        pred_student_output_img = netD(student_output_img)
        loss_G_GAN = torch.nn.MSELoss()(pred_student_output_img, target_real)

        # sparse loss
        loss_G_sparse, sparse_loss_dict, sparse_loss_dict_index = BIG_loss(netG)
        alpha = args.alpha*(1+2*i/args.epochs)

        # Total G loss
        loss_G = args.beta * loss_G_perceptual + loss_G_GAN + alpha * loss_G_sparse
        loss_G.backward()

        optimizer_G.step()
        optimizer_gamma.step()

        # append loss:
        loss_G_meter.append(loss_G.item())
        loss_G_perceptual_meter.append(loss_G_perceptual.item())
        loss_G_GAN_meter.append(loss_G_GAN.item())

        # proximal gradient for channel pruning:
        current_lr = lr_scheduler_gamma.get_lr()[0]
        # for name, m in netG.named_modules():
        #     if isinstance(m, nn.InstanceNorm2d) and m.weight is not None:
        #         m.weight.data = soft_threshold(m.weight.data, th=float(args.rho) * float(current_lr))
        
        if i % 10 == 0:
            print(i,'importants:',sparse_loss_dict[2],sep=',', file=file_handle1)
            print(i,'indexs:',sparse_loss_dict_index[2],sep=',', file=file_handle1)
        update_in(netG, sparse_loss_dict, sparse_loss_dict_index)

        if i % 50 == 0:
            if args.lc == 'vgg':
                out_str_G = 'epoch %d-%d-G: perceptual %.4f (content %.4f, style %.4f) | gamma lr %.4f | sparse loss %.4f' % (
                    epoch, i, loss_G_perceptual.data, loss_G_content.data, loss_G_style.data * 1e5, current_lr, loss_G_sparse.data)
            elif args.lc == 'mse':
                out_str_G = 'epoch %d-%d-G: mse %.4f | gamma lr %.4f | sparse loss %.4f' % (
                    epoch, i, loss_G_perceptual.data, current_lr, loss_G_sparse.data)
            print(out_str_G)
        ###### End G ######


        ###### D ######
        optimizer_D.zero_grad()

        # real loss:
        pred_teacher_output_img = netD(teacher_output_img)
        loss_D_real = torch.nn.MSELoss()(pred_teacher_output_img, target_real)

        # Fake loss
        student_output_img_buffer_pop = fake_img_buffer.push_and_pop(student_output_img)
        pred_student_output_img = netD(student_output_img_buffer_pop.detach())
        loss_D_fake = torch.nn.MSELoss()(pred_student_output_img, target_fake)

        # Total loss
        loss_D = (loss_D_real + loss_D_fake)*0.5
        loss_D.backward()

        optimizer_D.step()

        # append loss:
        loss_D_meter.append(loss_D.item())
        ###### End D ######


    ## at the end of each epoch
    netG.eval(), netG.eval()
    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()
    lr_scheduler_gamma.step()

    print('time: %.2f' % (time.time()-start_time))
    print(args)

    # plot training loss:
    losses = {}
    losses['loss_G'] = (loss_G_lst, loss_G_meter.avg)
    losses['loss_G_perceptual'] = (loss_G_perceptual_lst, loss_G_perceptual_meter.avg)
    losses['loss_G_GAN'] = (loss_G_GAN_lst, loss_G_GAN_meter.avg)
    losses['loss_D'] = (loss_D_lst, loss_D_meter.avg)
    for key in losses:
        losses[key][0].append(losses[key][1])
        plt.plot(losses[key][0])
        plt.savefig(os.path.join(results_dir, '%s.png' % key))
        plt.close()

    if epoch % 10 == 0 or epoch == args.epochs - 1:
        # save imgs:
        images={'input_img': input_img, 'teacher_output_img': teacher_output_img, 'student_output_img': student_output_img}
        for key in images:
            img_np = images[key].detach().cpu().numpy()
            img_np = np.moveaxis(img_np, 1, -1)
            img_np = (img_np + 1) / 2 # (-1,1) -> (0,1)
            img_big = fourD2threeD(img_np, n_row=4)
            print(key, img_big.shape, np.amax(img_big), np.amin(img_big))
            imsave(os.path.join(img_dir, 'epoch%d_%s.png' % (epoch, key)), img_as_ubyte(img_big))

    if epoch % 10 == 0 or epoch == args.epochs - 1:
        # Save models checkpoints
        g_path = os.path.join(pth_dir, 'epoch%d_netG.pth' % epoch)
        torch.save(netG.state_dict(), os.path.join(pth_dir, 'epoch%d_netG.pth' % epoch))
        # d_path = os.path.join(pth_dir, 'epoch%d_netD.pth' % epoch)
        # torch.save(netD.state_dict(), os.path.join(pth_dir, 'epoch%d_netD.pth' % epoch))
        result_dir = os.path.join(img_dir, 'epoch%d'%epoch)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        img_paths = get_image_file(args.test_image_dir)
        for img_path in img_paths:
            img_new = inference(img_path, netG)
            img_name = img_path.split('/')[-1]
            img_new_path = os.path.join(result_dir, img_name)
            cv2.imwrite(img_new_path, img_new)

        img_paths_for_FID_gt = [result_dir, args.image_gt_dir]
        # img_paths_for_FID_label = [args.image_fake_dir, args.image_label_dir]
        FID_gt = calculate_fid_given_paths(img_paths_for_FID_gt)
        # FID_label = calculate_fid_given_paths(img_paths_for_FID_label)
        FID_gt_lst.append(FID_gt)
        plt.plot(FID_gt_lst)
        plt.savefig(os.path.join(results_dir, 'fid_gt.png'))
        plt.close()
        # TBD: save best FID

    # channel number:
    channel_number_lst.append(none_zero_channel_num(netG))
    plt.plot(channel_number_lst)
    plt.savefig(os.path.join(results_dir, 'channel_number.png'))
    plt.close()

    # save latest:
    save_ckpt(epoch, netG, netD,
        optimizer_G, optimizer_D, optimizer_gamma,
        lr_scheduler_G, lr_scheduler_D, lr_scheduler_gamma,
        loss_G_lst, loss_G_perceptual_lst, loss_G_GAN_lst, loss_D_lst, channel_number_lst,
        path=os.path.join(pth_dir, 'latest.pth'))
###### End Training ######
