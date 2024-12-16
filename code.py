import cv2
import os
import joblib

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "models\haarcascade_frontalface_default.xml")
# Load the pre-trained KNN model
model_path = 'path_to_knn_model.pkl'  # Update with the correct model path
knn = joblib.load(model_path)
print("KNN model loaded successfully!")

# Function to preprocess face region for recognition
def preprocess_face(face_region):
    face_resized = cv2.resize(face_region, (100, 100))
    face_flattened = face_resized.flatten().reshape(1, -1)
    return face_flattened

# Function to predict the label using the KNN model
def recognize_face(face_region):
    face_data = preprocess_face(face_region)
    prediction = knn.predict(face_data)
    predicted_name = "Obama" if int(prediction[0]) == 0 else "Sorry not Obama!"
    return predicted_name

# To capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    ret, img = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        face_region = gray[y:y+h, x:x+w]

        # Recognize the face
        predicted_name = recognize_face(face_region)

        # Draw rectangle and annotate
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, predicted_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the frame with annotations
    cv2.imshow('Face Detection and Recognition', img)

    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(30) == 27:
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()

# Testing the model on new images
def test_on_images(test_folder):
    for filename in os.listdir(test_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            test_path = os.path.join(test_folder, filename)

            # Read and preprocess the image
            test_image = cv2.imread(test_path)
            gray_test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the test image
            faces = face_cascade.detectMultiScale(gray_test_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    face = gray_test_image[y:y+h, x:x+w]

                    # Recognize the face
                    predicted_name = recognize_face(face)

                    # Annotate the test image
                    cv2.putText(test_image, predicted_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.rectangle(test_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # Display the annotated image
                cv2.imshow('Test Image Recognition', test_image)
                cv2.waitKey(0)  # Wait for a key press before closing the image

            else:
                print(f"No faces detected in {filename}")

    cv2.destroyAllWindows()

# Example usage for testing on a folder
# test_folder = "/path_to_test_images"
# test_on_images(test_folder)
