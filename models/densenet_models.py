import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, ReLU, Dropout2d, MaxPool2d, ConvTranspose2d

class FCDenseNet(nn.Module):
    def __init__(self,
                 n_filters_first_conv = 48,
                 n_pool = 5,
                 growth_rate = 16,
                 dblock_layers = [4,5,7,10,12,15],# last one for denseblock
                 num_classes=2,
                 dropout_p=.2):
        super().__init__()
        
        # first conv layer
        self.conv1 = Conv2d(3, n_filters_first_conv, kernel_size=3, padding=1)
        
        n_filters = n_filters_first_conv

        # Downsampling path 
        lst_down_conv = nn.ModuleList()
        lst_down_td = nn.ModuleList()
        
        track_filters=[]
        for k in range(n_pool):
            lst_down_conv.append(DenseLayer(in_channels=n_filters, 
                                            growth_rate=growth_rate, 
                                            layers_per_block=dblock_layers[k], 
                                            dropout_p=dropout_p))
            n_filters += dblock_layers[k]*growth_rate
            
            lst_down_td.append(TransitionDown(in_channels=n_filters, out_channels=n_filters,dropout_p=dropout_p))
            
            track_filters.append(n_filters)
            
        self.down_convs_denseblock = lst_down_conv
        self.down_convs_td = lst_down_td
        
        # bottlencek layer
        self.bottleneck = DenseLayer(in_channels=n_filters, 
                                     growth_rate=growth_rate, 
                                     layers_per_block=dblock_layers[n_pool], 
                                     dropout_p=dropout_p)
        
        track_filters.append(n_filters+dblock_layers[n_pool]*growth_rate)
        track_filters = track_filters[::-1]
        n_filters+=dblock_layers[n_pool]*growth_rate
        
        flip_dblock_layers = dblock_layers[::-1]
        
        # Upsampling path
        lst_up_conv = nn.ModuleList()
        lst_up_tu = nn.ModuleList()
        
        for k in range(n_pool):
            lst_up_tu.append(TransitionUp(in_channels=n_filters, out_channels=growth_rate*flip_dblock_layers[k]))
            
            lst_up_conv.append(DenseLayer(in_channels=(track_filters[k+1]+ growth_rate*flip_dblock_layers[k]), 
                                            growth_rate=growth_rate, 
                                            layers_per_block=flip_dblock_layers[k+1], 
                                            dropout_p=dropout_p))
            n_filters = track_filters[k+1]+ growth_rate*flip_dblock_layers[k] + flip_dblock_layers[k+1]*growth_rate
            
            
        
        self.track_filters = track_filters
        self.up_convs_denseblock = lst_up_conv
        self.up_convs_tu = lst_up_tu
        
        
        # read out
        self.out = nn.Conv2d(n_filters,num_classes, kernel_size=(1,1))
        
    def forward(self,x):
        skip_connections = []
        
        x = self.conv1(x)
        
        for layer_dense, layer_td in zip(self.down_convs_denseblock,self.down_convs_td):
            x = layer_dense(x)
            #print('dense : ',x.shape)
            skip_connections.append(x)
            x = layer_td(x)
            #print('TD : ',x.shape)
            
        x = self.bottleneck(x)
        #print('bottleneck: ',x.shape)
        
        skip_connections = skip_connections[::-1]
        
        for idx, (layer_dense, layer_tu) in enumerate(zip(self.up_convs_denseblock,self.up_convs_tu)):
            x = layer_tu(x,skip_connections[idx])
            #print('skip : ',skip_connections[idx].shape)
            #print('up conv : ',x.shape)
            x = layer_dense(x)
            #print('down dense : ',x.shape)
        
        x = self.out(x)
        #print('readout : ',x.shape)
        return x

# initialization
def weights_init_norm(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)
        
def weights_init_he_uni(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
    #if isinstance(m, nn.BatchNorm2d):
    
class DenseLayer(nn.Module):
    def __init__(self, 
                 in_channels=48, 
                 growth_rate=12, 
                 layers_per_block=4,
                 dropout_p=0.1):
        super().__init__()
        
        lst=nn.ModuleList()
        self.layers_per_block=layers_per_block
        
        for _ in range(layers_per_block):
            lst.append(BN_ReLU_Conv(in_channels, growth_rate, kernel_size=(3,3), dropout_p=dropout_p))
            in_channels += growth_rate
        
        self.layers=lst
            
    def forward(self, x):
        
        for layer in self.layers:
            x = torch.cat([layer(x),x], dim = 1)
            
        return x
    
class TransitionUp(nn.Module):
    #skip_connection, block_to_upsample, n_filters_keep):
    """ Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection """
    def __init__(self, 
                 in_channels=48, 
                 out_channels=12):
        super().__init__()

        # Upsample
        self.layer = ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        
    def forward(self,x,skip_connection):
        x = self.layer(x)
        x = torch.cat([x, skip_connection], dim =1)
        
        return x

class TransitionDown(nn.Module):
    """ Apply first a BN_ReLu_conv layer with filter size = 1, and a max pooling with a factor 2  """
    def __init__(self, 
                 in_channels=48, 
                 out_channels=12,
                 dropout_p=.1):
        super().__init__()

        # Upsample
        self.layer = nn.Sequential(*BN_ReLU_Conv(in_channels, out_channels, kernel_size=(1,1), dropout_p=dropout_p),
                                   MaxPool2d(1, stride=2))
        
    def forward(self,x):
        x = self.layer(x)
        return x

def BN_ReLU_Conv(in_channels, out_channels, kernel_size=(3,3), dropout_p=0.2):
    """
    Apply successivly BatchNormalization, ReLu nonlinearity, Convolution and Dropout (if dropout_p > 0) on the inputs
    
    """
    padding = (kernel_size[0]-1)//2 if type(kernel_size)==tuple or type(kernel_size)==list else (kernel_size-1)//2
    
    lst=[BatchNorm2d(in_channels),
         ReLU(),
         Conv2d(in_channels,out_channels,kernel_size=kernel_size, stride=1, padding=padding)]
    
    if dropout_p > 0 : lst.append(Dropout2d(dropout_p))
    
    return nn.Sequential(*lst)
