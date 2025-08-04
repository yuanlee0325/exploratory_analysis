from skimage.measure import label
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import torch
from torch.nn import Sigmoid
from torchvision import transforms
from PIL import Image
import matplotlib.patches as mpatches
from time import time, sleep
from models.resnet_models import Generator



def get_trained_model(weight_path=r'D:\All_files\pys\AI_algos\Liv\TestMaskRCNN\Well96\weights\resnet_gen\gen_params_49_lr_1e-05.pt'):
    # get device
    device='cuda' if torch.cuda.is_available() else 'cpu'

    # get the model
    gen=Generator(num_residual_block=16,
                  residual_channels=64,
                  num_upscale_layers=1,
                  upscale_factor=1).to(device)
                  
    gen.load_state_dict(torch.load(weight_path,map_location=torch.device(device)))
    
    return gen


def get_inference(model,
                  image_path=r'D:\All_files\pys\AI_algos\Liv\TestMaskRCNN\Well96\Data\real_data\plate1_12w.png',
                  transform_hr=transforms.Compose([transforms.ToTensor(),transforms.Resize((512,512))]),
                  activation=Sigmoid(),
                  ):
    # get device
    device='cuda' if torch.cuda.is_available() else 'cpu'
    
    # getimage content
    img=Image.open(image_path)
    img_t=transform_hr(img)/255
    img_t=img_t.unsqueeze(0)
    del img

    # prediction
    mask_pred=model(img_t.to(device))
    dummy=activation(mask_pred[0].cpu().detach()).permute(1,2,0).numpy()
    dummy[dummy>.5]=1
    dummy[dummy<.5]=0
    dummy[dummy==.5]=1

    del mask_pred
    del model
    torch.cuda.empty_cache()

    return (img_t.squeeze(0).permute(1,2,0).numpy()*255,(dummy*255).astype('uint8'))
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
            if w*h>50 : cv2.rectangle(img, (x,y), (x+ w, y+h), (0,1,0), 1)

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
