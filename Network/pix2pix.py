import torch 
import torch.nn as nn
from torch.nn import init

def Defind_Gen(in_channels, out_channels, num_filters, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
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
        self.model = nn.Sequential()
        use_bias = 1

        self.model.append(nn.ReflectionPad2d(3))
        self.model.append(nn.Conv2d(in_channels, num_filters, kernel_size = 7, padding = 0))
        self.model.append(nn.BatchNorm2d(num_filters))
        self.model.append(nn.ReLU(True))

        n_downsampling = 2
        for i in range(n_downsampling): # DownSampling 
            multi = 2 ** i
            self.model.append(nn.Conv2d(multi * num_filters, multi * num_filters * 2, kernel_size = 3, stride = 2, padding = 1))
            self.model.append(nn.BatchNorm2d(multi * num_filters * 2))
            self.model.append(nn.ReLU(True))

        multi = 2 ** n_downsampling

        for i in range(num_blocks):   # Add ResnetBlock
            self.model.append(ResnetBlock(multi * num_filters, padding_type = padding_type, use_bias = use_bias, use_dropout = use_dropout))

        for i in range(n_downsampling): # UpSampling
            multi = 2 * (n_downsampling - i)
            self.model.append(nn.ConvTranspose2d(multi * num_filters, multi * num_filters / 2 , kernel_size = 3, 
                                    stride = 2,padding = 1))
            self.model.append(nn.BatchNorm2d(nn.BatchNorm2d(multi * num_filters / 2)))
            self.model.append(nn.ReLU(True))
        
        self.model.append(nn.ReflectionPad2d(3))
        self.model.append(nn.Conv2d(num_filters, out_channels, kernel_size = 7))
        self.model.append(nn.Tanh())

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

        conv = nn.Sequential()

        zero_padding = 0
        if padding_type == 'reflect':
            conv.append(nn.ReflectionPad1d(1))
        if padding_type == 'replicate':
            conv.append(nn.ReplicationPad1d(1))
        if padding_type == 'zero':
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
            conv.append(nn.ReflectionPad1d(1))
        if padding_type == 'replicate':
            conv.append(nn.ReplicationPad1d(1))
        if padding_type == 'zero':
            zero_padding = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        
        conv.append(nn.Conv2d(dim,dim,kernel_size=3,padding=zero_padding,bias=use_bias))
        conv.append(nn.BatchNorm2d(dim))

        return conv

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class  NLayerDiscriminator(nn.Module):
    """
    NLayerDiscriminator 
    Parameters:
        in_channels (int)       -- the number of channels in input images
    """
    def __init__(self,in_channels,):
        super(NLayerDiscriminator,self).__init__()
        self.network = nn.Sequential()
        self.network.append(nn.Conv2d(in_channels,64,
                            kernel_size = 4,stride = 2,padding=1))
        self.network.append(nn.LeakyReLU(0.2))
        
        last_chanel = 64

        for _ in range(3):
            self.network.append(nn.Conv2d(last_chanel, last_chanel*2, 
                                kernel_size= 4, stride= 2, padding= 1))
            self.network.append(nn.BatchNorm2d(last_chanel*2))
            self.network.append(nn.LeakyReLU(0.2))
            last_chanel = in_channels * 2

        self.network.append(nn.Conv2d(last_chanel,1,kernel_size= 4,
                            stride=2,padding=1))
        self.network.append(nn.Sigmoid())
    
    def forward(self,input):
        return self.network(input)

def Defind_Dis(in_channels, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = NLayerDiscriminator(in_channels)
    return init_net(net, init_type, init_gain, gpu_ids)

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