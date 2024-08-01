import cv2

def main():
    try:
        # Load the cascade classifiers for face and eye detection
        face_cascade = cv2.CascadeClassifier("Face Detection And Eye Detection\\Face-Eye-Detection-Opencv\\haarcascade_frontalface_default.xml")
        eye_cascade = cv2.CascadeClassifier("Face Detection And Eye Detection\\Face-Eye-Detection-Opencv\\haarcascade_eye_tree_eyeglasses.xml")
        
        # Check if the cascades have been loaded properly
        if face_cascade.empty():
            raise IOError("Error loading face cascade classifier xml file.")
        if eye_cascade.empty():
            raise IOError("Error loading eye cascade classifier xml file.")

        # Capture frames from the default camera (0)
        cap = cv2.VideoCapture(0)

        # Check if the camera opened successfully
        if not cap.isOpened():
            raise IOError("Error: Could not open video stream from camera.")

        # Loop to continuously get frames
        while True:
            # Read frame-by-frame
            ret, img = cap.read()
            
            # If frame reading is not successful
            if not ret:
                raise IOError("Error: Could not read frame from camera.")

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Loop through all detected faces
            for (x, y, w, h) in faces:
                # Draw rectangle around the face
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
                
                # Region of interest for detecting eyes within the face
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w] 
                
                # Detect eyes within the face region
                eyes = eye_cascade.detectMultiScale(roi_gray)
                
                # Draw rectangle around each eye
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 127, 255), 2)
            
            # Display the resulting frame
            cv2.imshow('img', img)
            
            # Wait for 'Esc' key to exit
            if cv2.waitKey(5) & 0xFF == 27:
                break

        # Release the capture and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

    except IOError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
