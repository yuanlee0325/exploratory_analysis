import torch
from models.densenet_models import FCDenseNet
import os
from torchvision import transforms
from PIL import Image
import cv2


def get_trained_densenet(model_name='densenet',
                     path=r'D:\All_files\pys\AI_algos\Liv\TestMaskRCNN\Well96\weights\densenet_gen',
                     params='unet_params_5_lr_0.001.pt'):
                     
    device='cuda' if torch.cuda.is_available() else 'cpu'
    
    # get the model
    gen = FCDenseNet(n_filters_first_conv = 48, 
                     n_pool = 5,
                     growth_rate = 16,
                     dblock_layers = [4,5,7,10,12,15],
                     num_classes=2).to(device)

    gen.load_state_dict(torch.load(os.path.join(path,params),map_location=torch.device(device)))
    
    return gen


def get_inference(model,
                  image_path=r'D:\All_files\pys\AI_algos\Liv\TestMaskRCNN\Well96\Data\real_data\plate1_12w.png',
                  transform_hr=transforms.Compose([transforms.ToTensor(),transforms.Resize((512,512))]),
                  ):
  
    # get device
    device='cuda' if torch.cuda.is_available() else 'cpu'
    
    # getimage content
    img=Image.open(image_path)
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