import os
import pandas as pd
import cv2

def extract_video_features(video_path):
    """
    Given the path to a video file, this function returns the total duration (in seconds)
    and the total number of frames in the video.
    """
    # Open the video using OpenCV
    video_capture = cv2.VideoCapture(video_path)
    
    if not video_capture.isOpened():
        raise FileNotFoundError(f"Video file {video_path} not found or cannot be opened.")
    
    # Get total frames and frame rate
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)  # Frames per second

    # Calculate total seconds (duration) of the video
    total_seconds = total_frames / frame_rate if frame_rate else 0
    
    # Release the video capture object
    video_capture.release()

    return total_seconds, total_frames

def process_excel_and_videos(excel_path, videos_path):
    """
    This function processes the Excel file, adds columns for the total duration and total frames of each video.
    A new copy of the Excel file is saved with these columns.
    """
    # Load the Excel file into a DataFrame
    df = pd.read_excel(excel_path, engine='openpyxl')
    
    # Initialize lists for the new columns
    total_seconds_list = []
    total_frames_list = []
    
    # Process each row in the Excel file
    for index, row in df.iterrows():
        video_id = row['VideoID']  # Assuming 'Video ID' is the column name for video IDs
        video_file = os.path.join(videos_path, f"{video_id}")  # Adjust the file extension if needed
        
        if os.path.exists(video_file):
            try:
                total_seconds, total_frames = extract_video_features(video_file)
                total_seconds_list.append(total_seconds)
                total_frames_list.append(total_frames)
            except FileNotFoundError:
                total_seconds_list.append(None)
                total_frames_list.append(None)
        else:
            total_seconds_list.append(None)
            total_frames_list.append(None)
    
    # Add new columns to the DataFrame
    df['Total Seconds'] = total_seconds_list
    df['Total Frames'] = total_frames_list
    
    # Save the modified DataFrame as a new Excel file
    new_excel_path = excel_path.replace('.xlsx', '_modified.xlsx')
    df.to_excel(new_excel_path, index=False, engine='openpyxl')
    
    return new_excel_path


# Example usage
excel_path = '/Users/jeremiah/phd_thesis/data/main_data/excel2.xlsx'  # Path to your original Excel file
videos_path = '/Users/jeremiah/phd_thesis/data/main_data/videos'  # Folder where the video files are located

new_excel_path = process_excel_and_videos(excel_path, videos_path)
print(f"The new Excel file is saved as: {new_excel_path}")
