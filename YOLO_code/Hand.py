import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient

# Initialize the InferenceHTTPClient
CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key="p3yTi4Yh626SgfhRYDKq"
)

# Define the model ID
MODEL_ID = "hand-segmentation-dne96/1"

def infer_frame(frame):
    # Convert the frame to the format expected by the inference client
    frame_path = 'temp_frame.jpg'
    cv2.imwrite(frame_path, frame)
    
    # Perform inference
    result = CLIENT.infer(frame_path, model_id=MODEL_ID)
    
    return result

def main():
    # Open the webcam
    cap = cv2.VideoCapture(2)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Perform inference on the frame
        result = infer_frame(frame)
        
        # Process and display the result
        hand_detected = False
        if 'predictions' in result:
            for prediction in result['predictions']:
                mask = prediction.get('mask')
                if mask is not None:
                    hand_detected = True
                    mask_array = np.array(mask, dtype=np.uint8)
                    # Ensure mask is same size as frame
                    mask_array = cv2.resize(mask_array, (frame.shape[1], frame.shape[0]))
                    # Create a colored overlay
                    colored_mask = np.zeros_like(frame)
                    colored_mask[mask_array > 0] = [0, 255, 0]  # Green color for mask
                    # Blend the colored mask with the frame
                    frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)
        
        if hand_detected:
            print("hand detect")
        
        # Display the frame with the inference results
        cv2.imshow('YOLO v8 Inference', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
