import sys
import os
# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.inference_unet import*
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import torch
from torchvision import transforms
from scipy import ndimage
from skimage.color import rgb2lab
import csv
import glob
from matplotlib.colors import LinearSegmentedColormap

model = get_trained_unet(model_name='unet_vgg11_upsampling',
                         path='./weights/',
                         params='unet_params_104_lr_1e-05_h2o2.pt')


def get_inference(model,
                  image_path,
                  transform_hr=transforms.Compose([transforms.ToTensor(),transforms.Resize((512,512), antialias=True)]),
                  crop = 0.2,
                  device = None
                  ):
  
    # get devic
     #if not(device):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    
    # getimage content
    img=Image.open(image_path)
    
    
    if crop :
        w, h = img.size
        left = int(w*crop)
        right = int(w - w*crop)
        top = int(h*crop)
        bottom = int(h - h*crop)
        img = img.crop((left,top,right, bottom))
        
    img_t=transform_hr(img)/255
    img_t=img_t.unsqueeze(0)
    del img

    # prediction
    model.eval()
    with torch.no_grad():
        mask_pred = model(img_t.to(device))
        
    _, mask_pred = torch.max(mask_pred, dim=1)
    
    mask = mask_pred.detach().cpu().squeeze(0).numpy()
    del mask_pred
    del model
    torch.cuda.empty_cache()

    return (img_t.squeeze(0).permute(1,2,0).numpy()*255, (mask*255).astype('uint8'))
    # convert single image and store in 


def squeeze_mask(image, mask, fac):
    '''
    reduces mask area by shortening ellipses mmajor and minor axis
    
    '''
    
    # fit BBoxes
    contours = cv2.findContours(mask,  cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    bboxes = []
    result = np.zeros(image.shape).astype('uint8')
    
    for i in contours:
        x,y,w,h = cv2.boundingRect(i)
        if w*h>30 : 
            bboxes.append([x,y,w,h])

    bboxes = np.array(bboxes)
    bboxes[:,0] = bboxes[:,0] + bboxes[:,2]/2
    bboxes[:,1] = bboxes[:,1] + bboxes[:,3]/2
    bboxes[:,2] = (1-fac)*bboxes[:,2]/2
    bboxes[:,3] = (1-fac)*bboxes[:,3]/2

    for el in bboxes:
        cv2.ellipse(result, (el[0],el[1]), (el[2],  el[3]), 0, 0,360, [255,255,255], -1)
    
    mask_orig = mask
    mask = result[:,:,0]

    return mask_orig, mask


def get_instance_masks(img, mask) -> np.ndarray:
    
    label_im, nb_labels = ndimage.label(mask) 

    instance_mask = np.zeros((*img.shape[:2],nb_labels))
    for i in range(nb_labels):

        mask_compare = np.full(np.shape(label_im), i+1) 
        separate_mask = np.equal(label_im, mask_compare).astype(int) 

        separate_mask[separate_mask == 1] = 1

        instance_mask[:,:,i] = separate_mask
            
    #assert instance_mask.shape[-1] == np.array(self.col_list).sum() , 'col_list does not match with instance mask dimension' 
    
    assert np.mod(instance_mask.shape[-1],8) == 0 , 'instance masks are not integer multiple of 8!' 
    
    col_list = [8]*(instance_mask.shape[-1]//8)
    
    print('col list modified : {}'.format(col_list) +'\n')

    return instance_mask, col_list


def sort_instance_masks(instance_mask, col_list, img) :
    '''
    takes instance mask and group mask accoriding to col_list. Use bounding box estimates on each masks 
    for sorting.
    args :
     - instance_mask : ndarray representing num wells containing liquids
     - col_list : group columns 

    reuturns :
     - instance_mask_sorted : instance sorted and grouped
     - bboxes_sorted : sorted bounding boxes

    '''
    
    
    lst_bbox = [] # bbox for sorting
    for i in range(instance_mask.shape[2]):
        separate_mask = instance_mask[:,:,i]
        lst_bbox.append(np.array(get_bboxes(img,separate_mask.astype('uint8'))).ravel())

    dummy =[]
    dummy_ids =[]
    for idx, i in enumerate(lst_bbox):
        if i.size > 0 :
            if i[2]*i[3]>100 and i.size<5:
                dummy.append(i)
                dummy_ids.append(idx)

    lst_bbox = dummy
    instance_mask= instance_mask[:,:,dummy_ids]

    lst_bbox = np.array(lst_bbox)

    idx = [idx for idx, el in enumerate(lst_bbox) if el.any()]
    instance_mask = instance_mask[:,:,idx]
    lst_bbox = np.array([lst_bbox[k].tolist() for k in idx])

    # x-sort
    idx = lst_bbox[:,0].argsort()
    lst_bbox = lst_bbox[idx]

    instance_mask = instance_mask[:,:,idx]

    # ysort provided by user
    bboxes_sorted =[]
    instance_mask_sorted = []
    start = 0
    for num in col_list:
        bboxes_sorted.append(lst_bbox[start:start+num])
        instance_mask_sorted.append(instance_mask[:,:, start:start+num])
        start += num

    # sort y
    #bboxes_sorted2 =#lst_sorted2 =[]
    for i, (el,els) in enumerate(zip(bboxes_sorted,instance_mask_sorted)):
        idx = el[:,1].argsort()
        bboxes_sorted[i] = el[idx]
        instance_mask_sorted[i] = els[:,:,idx]

    #print(bboxes_sorted)

    #instance_mask_sorted = instance_mask_sorted
    #self.bboxes_sorted = bboxes_sorted

    return instance_mask_sorted, bboxes_sorted


def get_colors_from_patches(
                            img,
                            instance_mask_sorted,
                            mode = 'rgb',
                            background_rgb = [255,255,255],
                            background_std = np.array([1e-8,1e-8,1e-8]),
                            verbose = False):
    
    color_list =[]
    err_list = []
            
    errfn = lambda r,e1,b,e2 : np.sqrt((1/r)**2 * e1**2 + (1/b)**2 * e2**2)
    epsi = 1e-12

    img = (img * 255).astype('uint8')

    # lab processing
    if mode == 'lab':
        # lab processing
        if verbose : print('running lab mode')
        for i in instance_mask_sorted:
            mask = i
            dummy =[]
            dummy_err = []
            im1 = np.zeros(img.shape)
            for j in range(mask.shape[2]):
                im1[:,:,0]= mask[:,:,j] * img[:,:,0]
                im1[:,:,1]= mask[:,:,j] * img[:,:,1]
                im1[:,:,2]= mask[:,:,j] * img[:,:,2]

                lab = rgb2lab(im1[mask[:,:,j]>0].astype('uint8')).mean(axis = 0)
                err  = rgb2lab(im1[mask[:,:,j]>0].astype('uint8')).std(axis = 0)

                dummy.append(lab.tolist())
                dummy_err.append(err.tolist())

            color_list.append(dummy)
            err_list.append(dummy_err)

    else : # rgb-resolved processing
        if verbose : print('running rgb-resolved mode')

        for i in instance_mask_sorted:
            mask = i
            dummy =[]
            dummy_err = []
            im1 = np.zeros(img.shape)
            for j in range(mask.shape[2]):
                im1[:,:,0]= mask[:,:,j] * img[:,:,0]
                im1[:,:,1]= mask[:,:,j] * img[:,:,1]
                im1[:,:,2]= mask[:,:,j] * img[:,:,2]
                rgb = im1[mask[:,:,j]>0].mean(axis = 0)
                err = im1[mask[:,:,j]>0].std(axis = 0)

                dummy.append(np.log(background_rgb/(rgb+epsi)).tolist())
                dummy_err.append(errfn(rgb,err,background_rgb,background_std).tolist())

            color_list.append(dummy)
            err_list.append(dummy_err)  


    return color_list, err_list

# Note: only suitable for lab colour model, because it will lack some rgb information under mode 'rgb'

def analysis(image_path):
    """ 
    image_path is a directory containing images
    Takes in an RGB image of wells and computes instance masks for filled wells.
    Then computes mean and std lab color for each well.
    Deletes all images in the directory except the last one.
    out is rows x columns x LAB
    """
    # Get a sorted list of all image path
    images = sorted(glob.glob(os.path.join(image_path, '*.png')))

    final_color_list = None
    final_error_list = None

    # Check if there are no images
    if not images:
        print("No images found in the directory.")
        return None, None

    # Loop over the images and process each one
    for i, image in enumerate(images):
        print(f'Processing image {i + 1}/{len(images)}: {image}')

        # Process each image
        img, mask = get_inference(model, image)
    
        orig_mask, squeezed_mask = squeeze_mask(img, mask, 0.4)
    
        instance_mask, col_list = get_instance_masks(img, squeezed_mask)
    
        im_sorted, bb_sorted = sort_instance_masks(instance_mask, col_list, img)

        color_list, err_list = get_colors_from_patches(img, im_sorted, 'lab')

        # Store the result of the last image
        if i == len(images) - 1:
            final_color_list = np.array(color_list)
            final_error_list = np.array(err_list)

        # Delete the image if it's not the last one
        if i < len(images) - 1:
            os.remove(image)
            print(f'Deleted image: {image}')

    return final_color_list, final_error_list
    

def show_images(image_path):
    """ 
    image_path a directory containing images

    """
    # Get a sorted list of all image path
    images = sorted(glob.glob(os.path.join(image_path, '*.png')))
    images2show =[]
    nrows, ncols = len(images), 1 # array of subplot
    figsize = [16,12] # figuresize, inches
    
    # create figure (fig), and array of axes (axs)
    fig , axs = plt.subplots(nrows=nrows,ncols=ncols, figsize=figsize)  
    grid_position = 0    
    for image in images:
        img=cv2.imread(image) 
        color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[grid_position].imshow(color)
        images2show.append(img)
        grid_position += 1
    print(f'The image_path contains {len(images)} images.\n')


def save_to_csv(color_list, err_list, csv_filename, folder):
    """
    Save the color list and error list to a CSV file with well information.
    
    Args:
        color_list (np.ndarray): Array containing LAB color information.
        err_list (np.ndarray): Array containing error information.
        csv_filename (str): The directory of the CSV file.
        folder: The file director, including images, analysis.csv
    """
    # Define row and column labels
    row_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    col_labels = list(range(1, 13))  # Columns 1-12
    # Flatten the data and prepare for CSV writing
    rows = []
    for row_idx, (color_row, err_row) in enumerate(zip(color_list, err_list)):
        for col_idx, (color, err) in enumerate(zip(color_row, err_row)):
            # Well information
            well_info = f"{row_labels[col_idx]}{col_labels[row_idx]}"
            # Combine well info, color, and error data into a single row
            row = [well_info] + color.tolist() + err.tolist()
            rows.append(row)

    # Define the header
    header = ['Well'] + ['L', 'A', 'B'] + ['Error_L', 'Error_A', 'Error_B']

    # Write the data to a CSV file
    folder = folder
    csv_filename= csv_filename
    file_path = os.path.join(folder, csv_filename) 
    try:
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)  # Write the header
            writer.writerows(rows)   # Write the rows
        print('CSV file successfully saved!')
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")
    
    
def sample_predict():
    pass
    # need to add this information
    # conversion = row ['degradation_rate']
    # replace this part with the developed Random Forest Classifier/Regressor


def sample_info_process(folder): 
    path_to_sample_records_xlsx = f'{folder}/methylene_blue_degradation.xlsx'
    path_to_analysis_csv= f'{folder}/analysis.csv' 
    # Load data from sample_records.xlsx
    sample_records_df = pd.read_excel(path_to_sample_records_xlsx)  
    # Load analysis data
    analysis_df = pd.read_csv(path_to_analysis_csv, encoding='ISO-8859-1')
    analysis_df.columns = analysis_df.columns.str.strip()  # Clean column names
    print("Analysis Columns:", analysis_df.columns)

    sample_dict = {}
    for _, row in sample_records_df.iterrows():
        well = row['Well']
        sample_dict[well] = {
            'sample_id': row.get('sample_id', 'N/A'),
            'catalyst_type': row.get('catalyst_type', 'N/A'),
            'chemical_component': row.get('chemical_component', 'N/A'),
            'light_source': row.get('light_source', 'N/A'),
            'wavelength': row.get('wavelength', 'N/A'),
            'time': row.get('irradiation_time', 'N/A'),
            'mass': row.get('mass_of_catalyst', 'N/A'),
            'solution': row.get('solution', 'N/A'), 
            'volume': row.get('volume', 'N/A'),  

            'atomsphere': row.get('atomsphere', 'N/A')
        }
    print("Sample Dictionary Contents:")
    print(sample_dict)  # Print the contents of the sample dictionary

    # Define row and column labels 
    row_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    col_labels = list(range(1, 13)) 

    # Create an empty DataFrame for the plate layout
    plate_df = pd.DataFrame(np.nan, index=row_labels, columns=col_labels)

    # Iterate through the analysis data and fill the DataFrame with both sample information and its corresponding result   
    for index, row in analysis_df.iterrows():
        well = row['Well']
        conversion = row ['degradation_rate']

        # Get the corresponding sample details from the sample records.csv
        sample_info = sample_dict.get(well, {})
        # Format the entry with the reaction conditions and result
        entry = (f'sample_id: {sample_info.get("sample_id", "N/A")}, '
                 f'catalyst_type: {sample_info.get("catalyst_type", "N/A")}, '
                 f'chemical_component: {sample_info.get("chemical_component", "N/A")}, '
                 f'light_source: {sample_info.get("light_source", "N/A")}, '
                 f'wavelength: {sample_info.get("wavelength", "N/A")} nm, '
                 f'time: {sample_info.get("time", "N/A")}, '
                 f'mass: {sample_info.get("mass", "N/A")} mg, '
                 f'volume: {sample_info.get("volume", "N/A")} uL, '
                 f'solution: {sample_info.get("solution", "N/A")}, '
                 f'atomsphere: {sample_info.get("atomsphere", "N/A")}, '
                 f'conversion: {conversion:.1f}' if conversion != 'N/A' else 'N/A')
        row_label = well[0]
        col_label = int(well[1:])
        plate_df.loc[row_label, col_label] = entry
    output_excel_path = os.path.join(folder, 'sample_results.xlsx')
    print(plate_df)
    plate_df.to_excel(output_excel_path)
    print(f"Data with sample details and results saved to: {output_excel_path}")
    visualize_plate(plate_df, folder)


def visualize_plate(plate_df, folder):
    # Extract conversion values from the results in the plate_df
    conversion_values = plate_df.applymap(lambda x: float(x.split('conversion: ')[-1]) if isinstance(x, str) and 'conversion: ' in x else np.nan)
    
    # Set up the figure and axis for plotting the 96-well grid
    fig, ax = plt.subplots(figsize=(18, 12))
    n_rows = 8  # A-H (8 rows)
    n_cols = 12  # 1-12 (12 columns)
    
    # Plot grid lines to represent the wells
    for row in range(n_rows + 1):
        ax.plot([0, n_cols], [row, row], color='black', lw=1)
    for col in range(n_cols + 1):
        ax.plot([col, col], [0, n_rows], color='black', lw=1)

    # Set axis limits
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)

    # Invert the y-axis to match the natural layout of the plate (A at the top)
    ax.invert_yaxis()

    # Function to calculate a shade of blue based on conversion value
    def calculate_blue_shade(value, min_value, max_value):
        # Normalize the value between 0 and 1
        normalized_value = (value - min_value) / (max_value - min_value) if max_value != min_value else 0
        # Scale between light blue (173, 216, 230) for lower values and white (255, 255, 255) for higher values
        r = int(120 + (255 - 120) * normalized_value)
        g = int(179 + (255 - 179) * normalized_value)
        b = int(240 + (255 - 240) * normalized_value)
        return (r / 255, g / 255, b / 255)

    # Get the min and max conversion values for normalization
    min_conversion = conversion_values.min().min()
    max_conversion = conversion_values.max().max()

    # Create a custom colormap that transitions from light blue to white
    colors = [(120 / 255, 179 / 255, 240 / 255), (1, 1, 1)]  # Light blue to white
    cmap = LinearSegmentedColormap.from_list("custom_blue_white", colors)

    # Add well labels and data to each grid cell, and color map based on conversion values
    for row_idx, row_label in enumerate(plate_df.index):
        for col_idx, col_label in enumerate(plate_df.columns):
            well_data = plate_df.loc[row_label, col_label]

            # Get the conversion value for this well
            conversion_value = conversion_values.loc[row_label, col_label]

            # Set the color based on the manually calculated blue shade
            if not np.isnan(conversion_value):
                color = calculate_blue_shade(conversion_value, min_conversion, max_conversion)
            else:
                color = (1, 1, 1, 1)  # Set color to white if no conversion value

            # Draw a rectangle for the well with the corresponding color
            rect = plt.Rectangle((col_idx, row_idx), 1, 1, facecolor=color, edgecolor='black')
            ax.add_patch(rect)

            # Parse the well data to extract chemical component and sample ID
            if isinstance(well_data, str):
                data_parts = well_data.split(', ')
                sample_id = next((part for part in data_parts if 'sample_id:' in part), 'N/A').replace('sample_id: ', '')
                chemical_component = next((part for part in data_parts if 'chemical_component:' in part), 'N/A').replace('chemical_component: ', '')
            else:
                sample_id = 'N/A'
                chemical_component = 'N/A'

            # Add well data (chemical component, sample_id, and conversion)
            ax.text(col_idx + 0.5, row_idx + 0.5, f'{sample_id}\n{chemical_component}\n{conversion_value:.1f}%',
                    va='center', ha='center', fontsize=10, color='black', fontname='Arial')

    ax.set_xticks([i + 0.5 for i in range(n_cols)])
    ax.set_xticklabels(range(1, n_cols + 1), fontsize=12)
    ax.set_yticks([i + 0.5 for i in range(n_rows)])
    ax.set_yticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], fontsize=12)
    ax.set_axis_off()

    # Add a colorbar for the conversion values that matches the custom colormap
    norm = plt.Normalize(vmin=min_conversion, vmax=max_conversion)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Add the colorbar to the plot
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', label='Degradation Rate (%)')
    cbar.set_label('Degradation Rate (%)', fontsize=12)

    # Save the plot as an image
    output_plot_path = f'{folder}/plate_visualization.png'
    plt.savefig(output_plot_path)
    print(f"Visualization saved to: {output_plot_path}")
    
