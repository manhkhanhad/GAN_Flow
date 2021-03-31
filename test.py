from PIL import Image
import tqdm 
from Network.pix2pix import Defind_Gen
from Network.pix2pix import Pix2Pix
from create_dataset import TestData
from Utils.Utils import setup_GPU, save_result
import os
import torch
import matplotlib.pyplot as plt

import argparse

if __name__== '__main__':
    #Define the common parserions
    parser = argparse.ArgumentParser()
    # basic parameters
    parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--direction', type=str, default='AtoB', help='direction "AtoB" or "BtoA"')
    # model parameters
    parser.add_argument('--model', type=str, default='pix2pix', help='chooses which model to use.')
    parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
    parser.add_argument('--netG', type=str, default='resnet_9', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
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
    # test parameters
    parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
    opt = parser.parse_args()

    dataset = TestData(opt)
    dataloader = torch.utils.data.DataLoader(
                                dataset,
                                batch_size=opt.batch_size,
                                shuffle= opt.shuffle,
                                num_workers=int(opt.num_threads))
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of test images = %d' % dataset_size)
    #model = Pix2Pix(opt)
    #model.load_networks('latest')
    setup_GPU(opt)

    Gen = Defind_Gen(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)

    #Load weight
    save_filename = 'latest_net_G.pth'
    save_dir = os.path.join(opt.checkpoints_dir,opt.name)
    save_path = os.path.join(save_dir, save_filename)
    print('loading the netG from %s' %save_path)
    if isinstance(Gen, torch.nn.DataParallel):
        Gen = Gen.module
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    state_dict = torch.load(save_path, map_location=str(device))
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    Gen.load_state_dict(state_dict)

    #Test
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            #model.set_input(data)  # unpack data from data loader
            input_image = data['test'].to(device)
            image_paths = data['test_paths']
            Gen.eval()
            result = Gen(input_image)
            
            #show test image
            #test_img = result[0]
            #print(test_img.shape)
            #test_img = test_img.permute(1,2,0)
            #print(test_img+1)
            #plt.imshow(test_img.cpu())
            #plt.imshow(test_img)

            #Save Image
            result_dir = os.path.join(opt.results_dir,opt.name)
            if not os.path.exists(result_dir):
                os.mkdir(result_dir)
            save_result(result,image_paths,result_dir)

            #Print process
            if i%5 == 0:
                print("[{}/{}]".format(i,len(dataset)))