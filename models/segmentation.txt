
from typing import List, Union, Tuple


def get_files(path : str, initials : str = None) -> List:
    '''
    return : function returns sorted file list
    args :
     - path : full path
     - initials : file name initials, e.g. for filename unitest_123454.png, use initials 'u'/'uni','unitest', etc.
    '''
    
    if initials: 
        if not(isinstance(initials, str)):
            raise TypeError('initials must be a string')
        
    os.chdir(path)
    ext = initials + '*.png' if initials else '*.png'
    file_list = np.array(glob(ext))
    #len(lst)

    # sort list
    sort_idx = np.array([int(file.split('.')[0].split('_')[-1]) for file in file_list]).astype('int')
    file_list = file_list[sort_idx.argsort()]
    
    return file_list


def sort_bboxes(img : np.ndarray , mask : List, col_list : List ) -> List:
    '''
    return : sorted bboxes
    args :
     - img : x * y * 3
     - mask : mask predictions from unet/densenet/resnet
     - col_list : user-defined list of columns
    '''
        
    # model preds
    bboxes = np.array(get_bboxes(img,mask))

    bboxes = bboxes[bboxes[:,0].argsort()] # sort x
    #col_list = [8,8] # col list provided by user
    bboxes_sorted =[]
    start = 0
    for num in col_list:
        bboxes_sorted.append(bboxes[start:start+num])
        start += num

    # sort y
    bboxes_sorted = [el[el[:,1].sort()].squeeze() for el in bboxes_sorted]
    return bboxes_sorted

def get_colors(img : np.ndarray,
               bboxes_sorted : List, 
               mode : str = 'rgb-resolved',
               crop : bool = False, 
               crop_ratio : List = [0.9, 0.9],
               background_rgb : Union[List,np.array] = [255,255,255]):

    color_list =[]
    err_list = []
    
    if not(background_rgb) : 
        background_rgb = img[img.shape[0]-10 :img.shape[0],img.shape[0]-10 :img.shape[0],:].mean(axis = (0,1))
    if not(isinstance(background_rgb,np.ndarray)) : background_rgb = np.array(background_rgb)
        

    # lab processing
    if mode == 'lab':
        # lab processing
        print('ruuning lab mode')
        for idx, k in enumerate(range(len(bboxes_sorted))):
            bbox = bboxes_sorted[idx]
            dummy = []
            err =[]
            for i in range(len(bbox)):
                x,y,w,h=bbox[i]

                if crop and np.array(crop_ratio).all():
                    x, y = int((x+w/2)), int((y+h/2))
                    woff, hoff = int(w/2 *crop_ratio[0]), int(h/2 *crop_ratio[1])
                    labs = rgb2lab((img[y-hoff:y+hoff,x-woff:x+woff,:]*255).astype('uint8'))
                else : 
                    labs = rgb2lab((img[y:y+h,x:x+w,:]*255).astype('uint8'))

                dummy.append(labs.mean(axis = (0,1)).tolist())
                err.append(labs.std(axis = (0,1)).tolist())
            color_list.append(dummy)
            err_list.append(err)
            
    else : # rgb-resolved processing
        print('running rgb-resolved mode')
        for idx, k in enumerate(range(len(bboxes_sorted))):
            bbox = bboxes_sorted[idx]
            dummy = []
            err =[]
            for i in range(len(bbox)):
                x,y,w,h=bbox[i]

                if crop and np.array(crop_ratio).all():
                    x, y = int((x+w/2)), int((y+h/2))
                    woff, hoff = int(w/2 *crop_ratio[0]), int(h/2 *crop_ratio[1])
                    rgbs = (img[y-hoff:y+hoff,x-woff:x+woff,:]*255).astype('uint8')

                else : 
                    rgbs = (img[y:y+h,x:x+w,:]*255).astype('uint8')

                dummy.append(np.log(background_rgb/rgbs.mean(axis = (0,1))).tolist())
                err.append(rgbs.std(axis = (0,1)).tolist())
            color_list.append(dummy)
            err_list.append(err)
            
    
            
    return color_list, err_list

def analyse_wells(file_list : Union[np.ndarray, List],
                  bboxes_sorted : List, 
                  mode : str = 'rgb-resolved',
                  crop : bool = False, 
                  crop_ratio : List = [0.9, 0.9],
                  background_rgb : Union[List,np.array] = [255,255,255],
                  chan : str  = 'g',
                  transform_hr  = transforms.Compose([transforms.ToTensor(),transforms.Resize((512,512))]),
                  verbose : bool = False
                 ):
    
    chan2id = {'r' : 0,'g': 1,'b': 2}
    
    for idx , file in enumerate(file_list):
        #print(file)
        img=Image.open(os.path.join(path,file)) #consisitent with torch's bilin interpolation
        img=transform_hr(img).permute(1,2,0).numpy()
        
        # single color
        colors, errs =  get_colors(img = img,
                                   bboxes_sorted = bboxes_sorted, 
                                   mode = mode,
                                   crop  = crop,
                                   crop_ratio = crop_ratio)
        if verbose: 
            # cropped patch plot
            plot_patches(img = img,
                     bboxes_sorted = bbox, 
                     crop = False,
                     crop_ratio=[0.9,0.9])


        #print(len(colors))
        if idx == 0 : dat = {k : [] for k in range(len(colors))}
        #print(dummy)

        for idx2, color in enumerate(colors):
            dat[idx2].append(np.array(color))
            
    return dat
    
def plot_patches(img : np.ndarray,
               bboxes_sorted : List,
                crop : bool = False, 
               crop_ratio : List = [0.9, 0.9]):
    # plotting
    for idx, k in enumerate(range(len(bboxes_sorted))):
        bbox = bboxes_sorted[idx]
        num_plots = len(bbox)
        fig ,ax = plt.subplots(1,num_plots,figsize=(5,3))
        for i in range(num_plots):
            x,y,w,h=bbox[i]
            if crop :
                x, y = int((x+w/2)), int((y+h/2))
                woff, hoff = int(w/2 *crop_ratio[0]), int(h/2 * crop_ratio[1])
                ax[i].imshow((img[y-hoff:y+hoff,x-woff:x+woff,:]*255).astype('uint8'))

            else:
                ax[i].imshow((img[y:y+h,x:x+w,:]*255).astype('uint8'))
            #ax[i].
            ax[i].axis('off')
        fig.suptitle('Plot column : '+str(idx))