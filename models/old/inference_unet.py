import torch
from models.unet_models import UNet, UNet_ups, UNet_vgg11
import os
from torchvision import transforms
from PIL import Image
import cv2

import matplotlib.patches as mpatches
from time import time, sleep
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import numpy as np


def get_trained_unet(model_name='unet',
                     path=r'D:\All_files\pys\AI_algos\Liv\TestMaskRCNN\Well96\weights\unet_gen',
                     params='unet_params_5_lr_0.001.pt',
                     device = None):
                     
    if not(device):
        device='cuda' if torch.cuda.is_available() else 'cpu'
    print('running on:',device)
    # get the model
    if model_name == 'unet' : gen= UNet().to(device)
    if model_name == 'unet_ups' : gen= UNet_ups().to(device)
    if model_name == 'unet_vgg11_transpose2d' : gen = UNet_vgg11( up_conv_layer='transpose2d').to(device)
    if model_name == 'unet_vgg11_upsampling' : gen = UNet_vgg11( up_conv_layer='upsampling').to(device)
    #params='gen_params_24_lr_0.0001.pt'

    gen.load_state_dict(torch.load(os.path.join(path,params),map_location=torch.device(device)))
    
    return gen


def get_inference(model,
                  image_path=r'D:\All_files\pys\AI_algos\Liv\TestMaskRCNN\Well96\Data\real_data\plate1_12w.png',
                  crop = 0.1,
                  transform_hr=transforms.Compose([transforms.ToTensor(),transforms.Resize((512,512))]),
                  device = None
                  ):
  
    # get devic
    if not(device):
        device='cuda' if torch.cuda.is_available() else 'cpu'
    
    # getimage content
    img=Image.open(image_path)
    if crop : img = crop_img(img,crop)
    img_t=transform_hr(img)/255
    img_t=img_t.unsqueeze(0)
    del img

    # prediction
    model.eval()
    with torch.no_grad():
        mask_pred=model(img_t.to(device))
        
    _, mask_pred = torch.max(mask_pred, dim=1)
    
    mask=mask_pred.detach().cpu().squeeze(0).numpy()
    del mask_pred
    del model
    torch.cuda.empty_cache()

    return (img_t.squeeze(0).permute(1,2,0).numpy()*255, (mask*255).astype('uint8'))
    # convert single image and store in 

def plot_single_image_inference(single_img,mask):
    while True:
        print(torch.randn(1))
        img=cv2.cvtColor(single_img,cv2.COLOR_BGR2RGB)

        contours = cv2.findContours(mask,  cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        for i in contours:
            x,y,w,h = cv2.boundingRect(i)
            #randnum=np.random.randint(0,10,4)
            #x,y,w,h=x+randnum[0],y+randnum[1],w+randnum[2],h+randnum[3]
            if w*h>80 : cv2.rectangle(img, (x,y), (x+ w, y+h), (0,0,0), 1)

        cv2.imshow("Bounding Rectangle", img)
        #cv2.imshow("Bounding Rectangle", thresh_image)
        key=cv2.waitKey(0)
        #cv2.destroyAllWindows()

        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
            
            
            
def get_bboxes(single_img,mask):
    img=cv2.cvtColor(single_img,cv2.COLOR_BGR2RGB)

    contours = cv2.findContours(mask,  cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    bboxes = []
    for i in contours:
        x,y,w,h = cv2.boundingRect(i)
        #randnum=np.random.randint(0,10,4)
        #x,y,w,h=x+randnum[0],y+randnum[1],w+randnum[2],h+randnum[3]
        if w*h>50 : 
        
            cv2.rectangle(img, (x,y), (x+ w, y+h), (0,1,0), 1)
            bboxes.append([x,y,w,h])
            
            
    return bboxes
            
            
def save_image(model,
               path=r'D:\All_files\pys\AI_algos\Liv\TestMaskRCNN\Well96\Data\real_data',
               file='plate1_12w.png',
               save_file_name = 'inference',
               bbox_edge_color = [0,1,0],
               bbox_edge_linewidth = 0.8):
    
    t1=time()
    (img,mask)=get_inference(model,
                             image_path=os.path.join(path,file),
                             transform_hr=transforms.Compose([transforms.ToTensor(),transforms.Resize((512,512))]))

    torch.cuda.empty_cache()

    # get bbox
    label_image=label(np.squeeze(mask))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img)

    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 100:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='green', linewidth=.8)
            ax.add_patch(rect)

    ax.set_axis_off()
    plt.tight_layout()
    #plt.show()
    plt.savefig(save_file_name+'_'+file,bbox_inches='tight')
    t2=time()
    print('elapsed time: {}'.format(t2-t1))
    
    
    
def crop_img(img,fac =0.1):
    (w,h) = img.size
    left = int(fac * w)
    right = int(w -fac*w)
    top = int(fac*h)
    bottom = int(h -fac*h)
    im2 = img.crop((left,top,right,bottom))
    return im2
    
