import torch 
import torch.nn as nn
from torch.nn import init
import os
from loss import GANLoss
from collections import OrderedDict


"""
*******************************************************************************************
*******************************************************************************************
**                                                                                       **
**                                      GENERATOR                                        **
**                                                                                       **
*******************************************************************************************
*******************************************************************************************
"""

def Defind_Gen(in_channels, out_channels, num_filters, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator
    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        num_filters (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks 
        norm (str) -- the name of normalization layers used in the network: batch | instance | none (may need later)
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns a generator
    """
    net = None
    if netG == 'resnet_6':
        net = ResnetGen(in_channels,out_channels,num_filters,num_blocks = 6, use_dropout = use_dropout)
    elif netG == 'resnet_9':
        net = ResnetGen(in_channels,out_channels,num_filters,num_blocks = 9, use_dropout = use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)
    

class ResnetGen(nn.Module):
    def __init__(self,in_channels, out_channels, num_filters , num_blocks, use_dropout,padding_type='reflect'):
        """
        Generator base on Resnet blocks:
        n_DownSampling --> n_Resnet Block --> n_UpSampling 

        Parameters:
            in_channels (int)       -- the number of channels in input images
            out_channels (int)      -- the number of channels in output images
            padding_type (str)      -- the name of padding layer in conv layers: reflect | replicate | zero
            num_filters (int)       -- the number of filters in the last conv layer
            num_blocks (int)        -- the number of ResNet blocks
            use_dropout (bool)      -- if use dropout layers
            
        """

        super(ResnetGen,self).__init__()
        self.model = []
        use_bias = 0
        self.model.append(nn.ReflectionPad2d(3))
        self.model.append(nn.Conv2d(in_channels, num_filters, kernel_size = 7, padding = 0, bias=use_bias))
        self.model.append(nn.BatchNorm2d(num_filters))
        self.model.append(nn.ReLU(True))

        n_downsampling = 2
        for i in range(n_downsampling): # DownSampling 
            multi = 2 ** i
            self.model.append(nn.Conv2d(multi * num_filters, multi * num_filters * 2, kernel_size = 3, stride = 2, padding = 1,bias=use_bias))
            self.model.append(nn.BatchNorm2d(multi * num_filters * 2))
            self.model.append(nn.ReLU(True))

        multi = 2 ** n_downsampling
        for i in range(num_blocks):   # Add ResnetBlock
            self.model.append(ResnetBlock(multi * num_filters, padding_type = padding_type, use_bias = use_bias, use_dropout = use_dropout))
        for i in range(n_downsampling): # UpSampling
            multi = 2 * (n_downsampling - i)
            self.model.append(nn.ConvTranspose2d(multi * num_filters, int(multi * num_filters / 2) , kernel_size = 3, 
                                    stride = 2,padding = 1,output_padding=1, bias=use_bias))
            self.model.append(nn.BatchNorm2d(int(multi * num_filters / 2)))
            self.model.append(nn.ReLU(True))
        
        self.model.append(nn.ReflectionPad2d(3))
        self.model.append(nn.Conv2d(num_filters, out_channels, kernel_size = 7,padding=0))
        self.model.append(nn.Tanh())

        self.model = nn.Sequential(*self.model)
        print(self.model)

    def forward(self, input):
        return self.model(input) 

        


class ResnetBlock(nn.Module):
    """ Define a Resmet Block """

    def __init__(self,dim,padding_type,use_bias,use_dropout):
        super(ResnetBlock,self).__init__()
        self.conv_block = self.buildResnetBlock(dim,padding_type,use_bias,use_dropout)
    
    def buildResnetBlock(self,dim,padding_type,use_bias,use_dropout):
        """ Build Block: Padding --> Conv2d --> Norm --> ReLU --> Padding --> Conv --> Norm 
        
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))

        """

        conv = []

        zero_padding = 0
        if padding_type == "reflect":
            conv.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            conv.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            zero_padding = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        
        conv.append(nn.Conv2d(dim,dim,kernel_size=3,padding=zero_padding,bias=use_bias))
        conv.append(nn.BatchNorm2d(dim))
        conv.append(nn.ReLU(True))

        if use_dropout:
            conv.append(nn.Dropout(0.5))

        zero_padding = 0
        if padding_type == 'reflect':
            conv.append(nn.ReflectionPad2d(1))
        elif padding_type == 'replicate':
            conv.append(nn.ReplicationPad2d(1))
        elif padding_type == 'zero':
            zero_padding = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        
        conv.append(nn.Conv2d(dim,dim,kernel_size=3,padding=zero_padding,bias=use_bias))
        conv.append(nn.BatchNorm2d(dim))

        return nn.Sequential(*conv)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

"""
*******************************************************************************************
*******************************************************************************************
**                                                                                       **
**                                      DISCRIMINATOR                                    **
**                                                                                       **
*******************************************************************************************
*******************************************************************************************
"""

class  NLayerDiscriminator(nn.Module):
    """
    NLayerDiscriminator 
    Parameters:
        in_channels (int)       -- the number of channels in input images
    """
    def __init__(self,in_channels,n_layers=3):
        super(NLayerDiscriminator,self).__init__()
        self.network = []
        self.network.append(nn.Conv2d(in_channels,64,
                            kernel_size = 4,stride = 2,padding=1))
        self.network.append(nn.LeakyReLU(0.2,True))
        
        use_bias = 0
        nf_mult = 1
        nf_mult_prev = 1
        ndf = 64
        for n in range(1, 3):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            self.network.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                                kernel_size= 4, stride= 2, padding= 1,bias=use_bias))
            self.network.append(nn.BatchNorm2d(ndf * nf_mult))
            self.network.append(nn.LeakyReLU(0.2,True))
        

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.network.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,kernel_size= 4,
                            stride=1,padding=1,bias=use_bias))
        self.network.append(nn.BatchNorm2d(ndf * nf_mult))
        self.network.append(nn.LeakyReLU(0.2,True))

        self.network.append(nn.Conv2d(ndf * nf_mult, 1,kernel_size= 4,
                            stride=1,padding=1))
        self.network = nn.Sequential(*self.network)
        print(self.network)
    def forward(self,input):
        return self.network(input)

def Defind_Dis(in_channels, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    Parameters:
        in_channels (int)     -- the number of channels in input images
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns a discriminator
    """

    net = NLayerDiscriminator(in_channels)
    return init_net(net, init_type, init_gain, gpu_ids)


"""
*******************************************************************************************
**                                                                                       **
**                                      UTIL FUNCTION                                    **
**                                                                                       **
*******************************************************************************************
"""



def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

def init_weights(net, init_type = 'normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, init_gain)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func> 


"""
*******************************************************************************************
**                                                                                       **
**                                      LOSS FUNCTION                                    **
**                                                                                       **
*******************************************************************************************
"""



"""
*******************************************************************************************
**                                                                                       **
**                                      Pix2Pix MODEL                                    **
**                                                                                       **
*******************************************************************************************
"""

class Pix2Pix():
    def __init__(self,opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'


        self.netG = Defind_Gen(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.netD = Defind_Dis(opt.input_nc + opt.output_nc, opt.init_type, opt.init_gain, self.gpu_ids)
            self.model_names = ['G', 'D']
            # define loss functions
            self.criterionGAN = GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        else:
            self.model_names = ['G']

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        fake_AB = torch.cat((self.real_A, self.fake_B),1) # Concat fake_B and real_A to feed to Discriminator
        pred_fake = self.netD(fake_AB.detach()) 
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        real_AB = torch.cat((self.real_A, self.real_B),1)
        pred_real = self.netD(real_AB.detach())
        self.loss_D_real = self.criterionGAN(pred_real,True)

        self.loss_D = 0.5 * (self.loss_D_fake + self.loss_D_real)
        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
    
    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate gradient for G
        self.optimizer_G.step()             # update G's weights

    def get_loss(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def set_input(self,input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
    
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def save_networks(self,epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if os.path.isdir(self.save_dir) == False:
                    os.makedirs(self.save_dir)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)
