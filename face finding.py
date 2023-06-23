import cv2

def detect_faces(image_path):
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over the detected faces and record the coordinates
    face_coordinates = []
    for (x, y, w, h) in faces:
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        face_coordinates.append((top_left, bottom_right))

    return face_coordinates

# Provide the path to your image
image_path = 'C:\\Users\\lunna1\\Pictures\\20200422_170911.jpg'

# Detect faces and get the coordinates
face_coordinates = detect_faces(image_path)

# Display the coordinates
for i, (top_left, bottom_right) in enumerate(face_coordinates):
    print(f"Face {i+1}:")
    print(f"  Top Left: {top_left}")
    print(f"  Bottom Right: {bottom_right}")
    print()
