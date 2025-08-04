import json
import os
import asyncio
import logging
from aiohttp import web
from pathlib import Path
from camera import Start_Realsense  # Import camera class 
from analysis.image_analysis import analysis, save_to_csv, sample_info_process # Import image analysis module 


# only check_file_status was added, reflect the analysis result and update move_commands.json could also be added
# use watchdog in opentrons protocol, create analysis.json in the server based on the analysis result

# Global variables
camera_instance = None
capture_completed = asyncio.Event()
OT2_HOSTNAME = "169.254.227.210"
ot2_client = None

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



async def capture_images(request):
    global camera_instance, capture_completed
    
    logger.info('Starting image capture process...')
    
    # Reset the event
    capture_completed.clear()

    # Load camera configuration from JSON file
    try:
        camera_path = Path('camera_config.json')
        if not camera_path.is_file():
            camera_path = Path(r"C:/Users/scrc112/Desktop/work/yuan/test/protocol/camera_config.json")
        with open(camera_path) as f:
            data = json.load(f)
        logger.info('Camera configuration loaded successfully.')
    
    except FileNotFoundError:
        logger.error('Camera configuration file not found.')
        return web.json_response({'status': 'Camera configuration file not found'}, status=500)

    except json.JSONDecodeError:
        logger.error('Error decoding JSON from the configuration file.')
        return web.json_response({'status': 'Invalid JSON in camera configuration file'}, status=500)

    # Initialize the camera object
    try:
        camera_instance = Start_Realsense(
            fname=data.get('fname', 'image_capture'),
            folder=data.get('folder', './captured_images'),
            frame_interval=data.get('frame_interval', 2.5),  # default 2.5 seconds if not provided
            stop=data.get('stop', 0.002),  # default 0.0025 hours (9 seconds) if not provided
            take_image=data.get('take_image', True),
            sensitivity=data.get('sensitivity', 110)
        )
        logger.info('Camera instance initialized successfully.')
    
    except Exception as e:
        logger.error(f'Error initializing camera: {e}')
        return web.json_response({'status': 'Error initializing camera'}, status=500)

    # Trigger image capture
    try:
        # Use run() method to handle image capture
        camera_instance.run()
        # Signal that capture is completed 
        capture_completed.set()       
        logger.info('Image capture completed.')
    
    except Exception as e:
        logger.error(f'Error during image capture: {e}')
        return web.json_response({'status': 'Error during image capture'}, status=500)

    # Analyze the images
    print('start to analyze ...')
    folder = data.get('folder')
    print(folder)
    try:
        color_list, err_list = analysis(image_path=folder)
        csv_filename = f'analysis.csv'
        save_to_csv(color_list, err_list, csv_filename, folder)
        logger.info(f'Images analysis completed. Results saved to {csv_filename}.')
    except Exception as e:
        logger.error(f'Error during image analysis: {e}')
        return web.json_response({'status': 'Error during image analysis'}, status=500)
    
    # Reset the camera instance after capturing
    camera_instance = None  
    logger.info('Camera instance reset after capturing.')

    return web.json_response({'status': 'Camera stopped and images analysed.'})


async def check_file_status(request):
    #Load camera configuration to get the folder path

    camera_path = Path('camera_config.json')
    if not camera_path.is_file():
        camera_path = Path(r"C:/Users/scrc112/Desktop/work/yuan/test/camera_config.json") 
    try:
        with open(camera_path) as f:
            data = json.load(f)
        folder = data.get('folder', './captured_images')
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return web.json_response({'status': 'error', 'message': f'Error loading camera configuration: {e}'}, status=500)
    filename = request.query.get('filename') 
    if not filename:
        return web.json_response({'status': 'error', 'message': 'Filename parameter is missing'}, status=400)
    # Construct the full file path
    file_path = os.path.join(folder, filename)  
    # Check if the analysis file exists
    if os.path.isfile(file_path):
        return web.json_response({'status': 'ready'})
    else:
        return web.json_response({'status': 'pending'}, status=404)


async def data_process(folder):
    camera_path = Path('camera_config.json')
    if not camera_path.is_file():
        camera_path = Path(r"C:/Users/scrc112/Desktop/work/yuan/test/camera_config.json") 
    try:
        with open(camera_path) as f:
            data = json.load(f)
        folder = data.get('folder')
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return web.json_response({'status': 'error', 'message': f'Error loading camera configuration: {e}'}, status=500)
    
    # Process sample information and visualize data in a plate layout
    sample_info_process(folder=folder) 
    image_path = f'{folder}/plate_visualization.png'
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return web.Response(body=image_data, content_type='image/png')


async def receive_message(request):
    """Receive and reflect messages sent from Opentrons."""
    data = await request.json()
    message = data.get('message', 'No message provided')
    print(f"Message received: {message}")
    return web.json_response({'status': 'message received', 'message': message})


async def handle_index(request):
    """Serve the index.html file."""
    try:
        with open('static/index.html', 'r', encoding='utf-8') as f:
            return web.Response(text=f.read(), content_type='text/html')
    except FileNotFoundError:
        return web.json_response({'status': 'index.html not found'}, status=404)
    except UnicodeDecodeError:
        return web.json_response({'status': 'error reading index.html'}, status=500)


async def init_app():
    app = web.Application()
    app.router.add_get('/', handle_index)  # Route for serving index.html
    app.router.add_get('/check_file_status', check_file_status) # check if the analysis file exists
    app.router.add_get('/data_process', data_process) # process b value and sample information
    app.router.add_post('/capture_images', capture_images) # capture images
    app.router.add_post('/send_message', receive_message)  # New route for receiving messages
    return app


if __name__ == '__main__':
    app = init_app()
    web.run_app(app, host='169.254.142.17', port=5000)