# this .py file is used for flexible data analysis, like selecting certain rows or columns # 
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import csv

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.image_analysis import analysis, show_images, save_to_csv, sample_info_process
import os




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
    
    ### Modify it to adapt to the machine learning model
    # Calculate the conversion percentage based on the equation y = y0 + A*exp(R0*x), y0 = -25.97383, A = 24.58419, R0 = -0.34525)
    df = pd.read_csv(file_path)
    b_values = df['B'].values # Convert the column to a numpy array
    # Define the function to calculate concemtration from b* value
    def calculate_x(y_values, y0, A, R0):
        x_values = []
        for y in y_values:
            if y > y0 + A * 1e-10:
                x = (1 / R0) * np.log((y-y0) / A)
                x_values.append(x)
            else:
                x = 20
                x_values.append(x)
        return x_values
    
    y0 = -25.97383
    A = 24.58419
    R0 = -0.34525

    # Calculate the corresponding x_values
    x_values = calculate_x(b_values, y0, A, R0)
    # Add the results to the Dataframe
    df['calculated_conc'] = x_values
    # Calculate the degradation rate using the formula(20-df['calculated_x])/20*100
    df['degradation_rate'] = (16- df['calculated_conc']) / 20 *100
    # Clip the degradation rate to ensure its between 0% and 100%
    df['degradation_rate'] = np.clip(df['degradation_rate'], 0, 100)
    # Save the updated Dataframe back to the same csv file
    df.to_csv(file_path, index = False)
    print(df.head())
    print('Successfully updated!')
 

def sample_info_process(folder): 
    path_to_sample_records_xlsx = f'{folder}/methylene_blue_degradation.xlsx'
    path_to_analysis_csv= f'{folder}/analysis.csv' 
    # Load data from sample_records.xlsx
    sample_records_df = pd.read_excel(path_to_sample_records_xlsx)  
    # Load analysis data
    analysis_df = pd.read_csv(path_to_analysis_csv)
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
        
        # Populate the Dataframe with the result

        row_label = well[0]
        col_label = int(well[1:])
        plate_df.loc[row_label, col_label] = entry
        
    # Save the populated DataFrame to an Excel file
    output_excel_path = os.path.join(folder, 'sample_results.xlsx')
    print(plate_df)
    plate_df.to_excel(output_excel_path)
    print(f"Data with sample details and results saved to: {output_excel_path}")
    visualize_plate(plate_df, folder)

# Visualization with color mapping
def visualize_plate(plate_df, folder):
    
    # Extract conversion values from the results in the plate_df
    conversion_values = plate_df.applymap(lambda x: float(x.split('conversion: ')[-1]) if isinstance(x, str) and 'conversion: ' in x else np.nan)
    
    # Set up the figure and axis for plotting the 96-well grid
    fig, ax = plt.subplots(figsize=(18, 12))
    # Number of rows and columns in a 96-well plate
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
                    va='center', ha='center', fontsize=8, color='black', fontname='Arial')

    ax.set_xticks([i + 0.5 for i in range(n_cols)])
    ax.set_xticklabels(range(1, n_cols + 1), fontsize=12)
    ax.set_yticks([i + 0.5 for i in range(n_rows)])
    ax.set_yticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], fontsize=12)

    # Remove axis borders
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
    
## process data in a specific order
#filtration_result = []
#for i in range(0,3): 
#    new_array = color_list[i][:4,:5,2] # 1st index: image number in the 'image_path' list; 2nd index: ncols, nrows, and lab color parameters
#    filtration_result.append(new_array)
#num1 = len(filtration_result)   
#a = filtration_result[0].T
#b = filtration_result[1].T
#c = filtration_result[2].T
#d_filter = (a + b + c) / num1 
#print(d_filter)#

#color_gradient = []
#for j in range(3,6):
#   new_array = color_list[j][4:6,:5,2] 
#    color_gradient.append(new_array)
#num2 = len(color_gradient)
#a = color_gradient[0].T
#b = color_gradient[1].T
#c = color_gradient[2].T
#d_gradient = (a + b + c) / num2 
#print(d_gradient)
# #save results, e.g. CSV 
#dataframe1 = pd.DataFrame(d_filter) 
#dataframe2 = pd.DataFrame(d_gradient)
#dataframe1.to_csv(r"C:/Users/scrc112/Desktop/work/yuan/20240801\data1.csv")
#dataframe2.to_csv(r"C:/Users/scrc112/Desktop/work/yuan/20240801\data2.csv")


if __name__ == '__main__':

    image_path = r'C:\Users\scrc112\Desktop\work\yuan\wha_20250203\preestimation' 
    csv_filename = 'analysis.csv'
    #show_images(image_path) # it only works for a series of images(>=2) 
    mean, std = analysis(image_path) # mean(x, y) or std(x, y), x = column, y = row, preprocessed mean and std due to the mask
    save_to_csv(mean, std, csv_filename, folder=image_path)
    #sample_info_process(folder=image_path)
