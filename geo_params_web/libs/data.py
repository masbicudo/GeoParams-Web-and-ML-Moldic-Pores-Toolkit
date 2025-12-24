import glob
import json
import os
import uuid
import pandas as pd
import cv2

from libs.images import binarize_c_k

# Function to check if a string is a valid GUID
def is_valid_guid(guid_str):
    try:
        uuid.UUID(guid_str)  # Try to create a UUID object from the string
        return True
    except ValueError:
        return False

def is_option_canceled(opts):
    canceled = opts.get("params_select.state", "") == "Cancel"
    if "region_select.state" in opts:
        canceled = canceled or opts.get("region_select.state", "") == "Cancel"
    return canceled

def is_option_test(opts):
    name = opts.get("user", {}).get("name", "")
    return "test" in name.casefold()

# Function to load data from JSON files
def load_data_from_files(debug=False, print=print):
    
    path_pattern = 'static/output/*/*/options.json'
    
    files = glob.glob(path_pattern)  # Match files with the given pattern
    data = []

    for file in files:
        if debug: print(f"file={file}")
        # Extract the GUID from the folder structure
        folder_name = os.path.basename(os.path.dirname(os.path.dirname(file)))
        
        # Check if the folder name is a valid GUID
        if not is_valid_guid(folder_name):
            continue  # Skip files in folders that don't have a valid GUID
        
        # Define the path for the corresponding end_review.txt
        review_file_path = os.path.join(os.path.dirname(file), 'end_review.txt')

        # Check if the review file exists
        review_text = ""
        if os.path.exists(review_file_path):
            try:
                with open(review_file_path, 'r', encoding="utf-8") as review_file:
                    review_text = review_file.read()
            except Exception as e:
                with open(review_file_path, 'r') as review_file:
                    review_text = review_file.read()

        with open(file, 'r') as f:
            json_data = json.load(f)
            name = json_data.get("user", {}).get("name")
            email = json_data.get("user", {}).get("email")
            anonymize = json_data.get("user", {}).get("anonymize")
            experience = json_data.get("user", {}).get("experience")
            filename = json_data.get("image_select.filename")
            min_pore_size = json_data.get("initial_image_setup.min_pore_size")
            clicked_points = json_data.get("params_select.clicked_points", [])
            canceled = is_option_canceled(json_data)

            # legacy support for old keys
            if "region_select.clicked_points" in json_data:
                clicked_points.extend(json_data.get("region_select.clicked_points", []))

            if debug: print(f"len(clicked_points)={len(clicked_points)}")

            # Store relevant data for each clicked point
            for point in clicked_points:
                d = {
                        'experience': experience,
                        'filename': filename,
                        'min_pore_size': min_pore_size,
                        'clicked_x': point['x']*8,
                        'clicked_y': point['y']*8,
                        
                        # data removed (set to null) for privacy concerns
                        'review': review_text,
                        'name': name,
                        'email': email,
                        'anonymize': anonymize,
                        
                        'folder_name': folder_name,
                        'canceled': canceled,
                    }
                if debug: print(f"d={d}")
                data.append(d)
    
    return pd.DataFrame(data)

def get_available_images(filter_user_name, debug=False):
    # This function is only used when participant-level views are enabled.
    # In public/anonymized mode, it should not be called.
    
    path_pattern = 'static/output/*/*/options.json'
    
    files = glob.glob(path_pattern)  # Match files with the given pattern
    
    result = []
    for file in files:
        if debug: print(f"file={file}")
        # Extract the GUID from the folder structure
        folder_name = os.path.basename(os.path.dirname(os.path.dirname(file)))
        
        # Check if the folder name is a valid GUID
        if not is_valid_guid(folder_name):
            continue  # Skip files in folders that don't have a valid GUID
        
        with open(file, 'r') as f:
            json_data = json.load(f)
        user_name = json_data["user"].get("name", "")
        
        if filter_user_name and user_name.lower() != filter_user_name.lower():
            continue

        # Define the path for the corresponding cropped.jpg
        cropped_image_path = os.path.join(os.path.dirname(file), 'cropped.jpg')
        cropped_image = cv2.imread(cropped_image_path)
        tile_height, tile_width = cropped_image.shape[:2]
        param_space_image_path = os.path.join(os.path.dirname(file), 'main_image.png')
        param_space_image = cv2.imread(param_space_image_path)
        tiles_image_path = os.path.join(os.path.dirname(file), 'stitched_tiles.png')
        tiles_image = cv2.imread(tiles_image_path)
        
        tiles_images = []
        user_info = {
            # data removed (set to null) for privacy concerns
            'name': user_name,
            
            'folder_name': os.path.dirname(file),
            'cropped_image': cropped_image,
            'param_space_image': param_space_image,
            'tiles_image': tiles_image,
            'tile_height': tile_height,
            'tile_width': tile_width,
            'tiles': tiles_images,
        }
        result.append(user_info)
        
        clicked_points = json_data.get("params_select.clicked_points", [])
        for point in clicked_points:
            x, y = point["x"], point["y"]
            tile_image = tiles_image[y*tile_height:(y+1)*tile_height, x*tile_width:(x+1)*tile_width]
            
            tiles_images.append({
                'x': x,
                'y': y,
                'image': tile_image,
            })
            
    return result
