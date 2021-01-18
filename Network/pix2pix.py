import torch 
import torch.nn as nn

class ResnetGen(nn.Module):
    def __init__(self,in_channels, out_channels, num_filters, padding_type, norm_layer, num_blocks, use_dropout):
        """
        Parameters:
            in_channels (int)       -- the number of channels in input images
            out_channels (int)      -- the number of channels in output images
            padding_type (str)      -- the name of padding layer in conv layers: reflect | replicate | zero
            num_filters (int)       -- the number of filters in the last conv layer
            norm_layer              -- normalization layer
            num_blocks (int)        -- the number of ResNet blocks
            use_dropout (bool)      -- if use dropout layers
            
        """

        super(ResnetGen,self).__init__()
        self.model = nn.Sequential()
        use_bias = 1

        self.model.append(nn.ReflectionPad2d(3))
        self.model.append(nn.Conv2d(in_channels, num_filters, kernel_size = 7, padding = 0))
        self.model.append(norm_layer)
        self.model.append(nn.ReLU(True))

        n_downsampling = 2
        for i in range(n_downsampling): # DownSampling 
            multi = 2 ** i
            self.model.append(nn.Conv2d(multi * num_filters, multi * num_filters * 2, kernel_size = 3, stride = 2, padding = 1))
            self.model.append(norm_layer)
            self.model.append(nn.ReLU(True))

        multi = 2 ** n_downsampling

        for i in range(num_blocks):   # Add ResnetBlock
            self.model.append(ResnetBlock(multi * num_filters, padding_type = padding_type, use_bias = use_bias, 
                                            norm_layer = norm_layer, use_dropout = use_dropout))

        for i in range(n_downsampling): # UpSampling
            multi = 2 * (n_downsampling - i)
            self.model.append(nn.ConvTranspose2d(multi * num_filters, multi * num_filters / 2 , kernel_size = 3, 
                                    stride = 2,padding = 1))
            self.model.append(norm_layer)
            self.model.append(nn.ReLU(True))
        
        self.model.append(nn.ReflectionPad2d(3))
        self.model.append(nn.Conv2d(num_filters, out_channels, kernel_size = 7))
        self.model.append(nn.Tanh())

    def forward(self, input):
        return self.model(input) 

        


class ResnetBlock(nn.Module):
    """ Define a Resmet Block """

    def __init__(self,dim,padding_type,use_bias,use_dropout,norm_layer):
        super(ResnetBlock,self).__init__()
        self.conv_block = self.buildResnetBlock(dim,padding_type,use_bias,use_dropout,norm_layer)
    
    def buildResnetBlock(self,dim,padding_type,use_bias,use_dropout,norm_layer):
        """ Build Block: Padding --> Conv2d --> Norm --> ReLU --> Padding --> Conv --> Norm 
        
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
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
        conv.append(norm_layer(dim))
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
        conv.append(norm_layer(dim))

        return conv

    def foward(self, x):
        out = x + self.conv_block(x)
        return out



