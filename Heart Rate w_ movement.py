import cv2
import numpy as np
import pandas as pd
import os
from scipy import signal

numFrames = 22  # Number of frames to process
Fs = 30  # Frame rate
frame_folder_path = "C:\\Users\\Lunna1\\Desktop\\FRAMES\\"  # Path to save frames
video_path = r"C:\Users\lunna1\Pictures\Camera Roll\WIN_20230621_13_13_03_Pro.mp4"
excel_file_path = r"C:\Users\lunna1\Desktop\Senior Design\MOVEMENT.xlsx"  # Path to the Excel file for storing coordinates

print("Processing frames...")
capture = cv2.VideoCapture(video_path)
df_coordinates = pd.DataFrame(columns=['Frame Number', 'Top Left', 'Bottom Right'])

# Function to find faces and store coordinates in Excel
def detect_faces(image_path, excel_file):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    face_coordinates = []
    for (x, y, w, h) in faces:
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        face_coordinates.append((top_left, bottom_right))

    # Update the DataFrame with face coordinates
    frame_number = int(os.path.basename(image_path).split('Image')[1].split('Frame')[0]) - 1
    df_coordinates.at[frame_number, 'Top Left X'] = face_coordinates[0][0][0]
    df_coordinates.at[frame_number, 'Top Left Y'] = face_coordinates[0][0][1]
    df_coordinates.at[frame_number, 'Bottom Right X'] = face_coordinates[0][1][0]
    df_coordinates.at[frame_number, 'Bottom Right Y'] = face_coordinates[0][1][1]

    # Save the updated DataFrame to Excel
    df_coordinates.to_excel(excel_file, index=False)
    
    # Return the face coordinates
    return face_coordinates


# Save the initial coordinates to the Excel file
df_coordinates.to_excel(excel_file_path, index=False)

def process_frame(frame, frame_number):
    face_coordinates = detect_faces(frame_folder_path + f'Image{frame_number}Frame.tiff', excel_file_path)
    # Detect faces in the frame
    if len(face_coordinates) > 0:
        # Crop the region containing the first detected face
        (x, y), (x2, y2) = face_coordinates[0]
        region = frame[y:y2, x:x2]
        G = np.array(region)[:, :, 1]
        mean_value = np.mean(G)
        return mean_value
    return 0

i = 1
while capture.isOpened() and i <= numFrames:
    ret, frame = capture.read()
    if not ret:
        break

    cv2.imwrite(frame_folder_path + f'Image{i}Frame.tiff', frame)
    print(f"Processed frame {i}/{numFrames}")

    # Detect faces and store coordinates in Excel
    mean_value = process_frame(frame, i)

    i += 1

capture.release()
cv2.destroyAllWindows()

print("Frame processing complete")

# Check if the actual number of frames is less than numFrames
if i - 1 < numFrames:
    numFrames = i - 1
    print(f"Actual number of frames: {numFrames}")

# Process frames
print("Preprocessing Images")
graphthis = np.zeros(numFrames)

# Process frames one by one
for frame_number in range(1, numFrames + 1):
    frame_path = frame_folder_path + f'Image{frame_number}Frame.tiff'
    if os.path.exists(frame_path):
        frame = cv2.imread(frame_path)
        mean_value = process_frame(frame, frame_number)
        graphthis[frame_number - 1] = mean_value

# Delete frames
print("Deleting frames...")
for i in range(1, numFrames + 1):
    namef = frame_folder_path + f'Image{i}Frame.tiff'
    if os.path.exists(namef):
        os.remove(namef)
print("Frames deleted")

normean = signal.detrend(graphthis)

L_out = len(normean) - 1
f = Fs * np.arange(L_out // 2 + 1) / L_out

n = 3
Rs = 80
Wn = [0.2, 5]
Ws = np.array(Wn) / (Fs / 2)
ftype = 'bandpass'

b, a = signal.cheby2(n, Rs, Ws, ftype, output='ba')
realout = signal.filtfilt(b, a, normean)
Cheb2 = np.fft.fft(realout)
no_filt = np.fft.fft(normean)

p4 = np.abs(no_filt)
p3 = p4[:L_out // 2 + 1]
p2 = np.abs(Cheb2)
p1 = p2[:L_out // 2 + 1]

peak_index = np.argmax(p1)
peak_bpm = f[peak_index] * 60
print(f"Peak BPM: {peak_bpm}")

# Load the movement Excel sheet
df_movement = pd.read_excel(excel_file_path)

# Calculate movement distance
movement_x = df_movement['Bottom Right X'] - df_movement['Top Left X']
movement_y = df_movement['Bottom Right Y'] - df_movement['Top Left Y']
movement_distance = np.sqrt(movement_x ** 2 + movement_y ** 2)

# Determine if the person was relatively still or actively moving
movement_threshold = 10  # Adjust this threshold as needed
is_still = np.all(movement_distance < movement_threshold)
is_moving = not is_still

# Create a new DataFrame to store the movement status and peak heart rate
df_status = pd.DataFrame({'Movement Status': [is_moving], 'Peak Heart Rate': [peak_bpm]})

# Save the movement status and peak heart rate to a new Excel file
status_file_path = r"C:\Users\lunna1\Desktop\Senior Design\Movement_Status.xlsx"
df_status.to_excel(status_file_path, index=False)
