import argparse
import time
from create_dataset import PairDataset
from Network.pix2pix import Pix2Pix
from Utils.setup import setup_GPU
import torch
from tqdm import tqdm
import os

if __name__== '__main__':
    #Define the common parserions
    parser = argparse.ArgumentParser()
    # basic parameters
    parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--direction', type=str, default='AtoB', help='direction "AtoB" or "BtoA"')
    parser.add_argument('--isTrain', type=str, default='True', help='Train model.')
    # model parameters
    parser.add_argument('--model', type=str, default='pix2pix', help='chooses which model to use.')
    parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
    parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
    parser.add_argument('--netG', type=str, default='resnet_9', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
    parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
    # dataset parameters
    parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--shuffle', type=int, default=False, help='shuffle dataset')
    # training parameters
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
    parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
    parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--lambda_L1',type=float, default=100,help="weight for L1 loss")
    parser.add_argument('--save_epoch_freq',type=int, default=5,help="weight for L1 loss")

    opt = parser.parse_args()
    #set up GPU
    setup_GPU(opt)
    #Load dataset
    dataset = PairDataset(opt)
    dataloader = torch.utils.data.DataLoader(
                                dataset,
                                batch_size=opt.batch_size,
                                shuffle= opt.shuffle,
                                num_workers=int(opt.num_threads))
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    #Create model
    model = Pix2Pix(opt)

    
    if opt.continue_train:
        model.load_networks('latest')
        path = os.path.join(opt.checkpoints_dir,opt.name)
        list_epoch = []
        for name in os.listdir(path):
            epoch = name[:name.find('_')]
            if epoch != 'latest':
                list_epoch.append(epoch)
        print(max(list_epoch))
        opt.epoch_count = int(max(list_epoch))

    total_iters = 0
    for epoch in tqdm(range(opt.epoch_count+1, opt.n_epochs + opt.n_epochs_decay + 1)):
        epoch_start = time.time()
        for i,data in enumerate(dataloader):
            iter_start_time = time.time()
            model.set_input(data)
            model.optimize_parameters()

        #Save model
        if epoch % opt.save_epoch_freq == 0:       
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks('latest')
        model.save_networks(epoch)
        #Print Loss
        loss = model.get_loss()
        time_per_epoch = time.time() - epoch_start
        message = '(epoch: %d/%d, time: %.3f) ' % (epoch, opt.n_epochs + opt.n_epochs_decay, time_per_epoch )
        for k, v in loss.items():
            message += '%s: %.3f ' % (k, v)
        print(message)




