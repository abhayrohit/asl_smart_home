import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import tensorflow as tf
import RPi.GPIO as GPIO  # For Raspberry Pi GPIO control

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Define pins for LED controls
LIGHT_PIN = 17  # LED representing light
FAN_PIN = 27    # LED representing fan
CURTAIN_PIN = 22  # LED representing curtain
GPIO.setup(LIGHT_PIN, GPIO.OUT)
GPIO.setup(FAN_PIN, GPIO.OUT)
GPIO.setup(CURTAIN_PIN, GPIO.OUT)

# Initialize all outputs to OFF
GPIO.output(LIGHT_PIN, GPIO.LOW)
GPIO.output(FAN_PIN, GPIO.LOW)
GPIO.output(CURTAIN_PIN, GPIO.LOW)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define smart home action mapping
ACTION_MAP = {
    'A': "Turn on lights",
    'B': "Turn off lights",
    'L': "Lock doors",
    'V': "Open curtains",
    'W': "Turn on fan",
    'Y': "Close curtains"
}

# Colors for visualization
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
PURPLE = (255, 0, 255)
ORANGE = (0, 165, 255)

def load_tflite_model(model_path='asl_model.tflite'):
    """Load the TFLite model"""
    try:
        # Use tflite_runtime if available (better for Raspberry Pi)
        try:
            import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(model_path=model_path)
        except ImportError:
            # Fall back to TensorFlow implementation
            interpreter = tf.lite.Interpreter(model_path=model_path)
            
        interpreter.allocate_tensors()
        print("TFLite model loaded successfully!")
        return interpreter
    except Exception as e:
        print(f"Error loading TFLite model: {e}")
        return None

def load_preprocessing_data():
    """Load the label encoder and normalization values"""
    try:
        # Load label encoder
        with open('asl_label_encoder.pickle', 'rb') as f:
            label_encoder = pickle.load(f)
            
        # Load normalization values
        with open('asl_normalization.pickle', 'rb') as f:
            norm_data = pickle.load(f)
            X_min = norm_data['min']
            X_max = norm_data['max']
            
        # Load class mapping
        with open('asl_class_mapping.pickle', 'rb') as f:
            class_mapping = pickle.load(f)
            
        return label_encoder, X_min, X_max, class_mapping
    except FileNotFoundError as e:
        print(f"Error loading preprocessing data: {e}")
        print("Make sure you have run the training script first.")
        return None, None, None, None
    except Exception as e:
        print(f"Error loading preprocessing data: {e}")
        return None, None, None, None

def process_input_data(landmarks, X_min, X_max):
    """Process input data for the TFLite model"""
    # Ensure we have the right number of landmarks
    if len(landmarks) < 42:
        return None
        
    # Normalize using the same parameters as during training
    landmarks_array = np.array(landmarks[:42], dtype=np.float32).reshape(1, 42)
    normalized = (landmarks_array - X_min) / (X_max - X_min + 1e-7)
    
    return normalized

def predict_with_tflite(interpreter, input_data):
    """Run prediction with TFLite model"""
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Get the predicted class
    predicted_class = np.argmax(output_data[0])
    
    return predicted_class, output_data[0][predicted_class]

def control_smart_home(gesture):
    """Control smart home devices based on gesture"""
    if gesture == 'A':  # Turn on lights
        GPIO.output(LIGHT_PIN, GPIO.HIGH)
        print("Lights ON")
    elif gesture == 'B':  # Turn off lights
        GPIO.output(LIGHT_PIN, GPIO.LOW)
        print("Lights OFF")
    elif gesture == 'W':  # Turn on fan
        GPIO.output(FAN_PIN, GPIO.HIGH)
        print("Fan ON")
    elif gesture == 'L':  # Turn off fan (repurposing Lock doors)
        GPIO.output(FAN_PIN, GPIO.LOW)
        print("Fan OFF")
    elif gesture == 'V':  # Open curtains
        GPIO.output(CURTAIN_PIN, GPIO.HIGH)
        print("Curtains OPEN")
    elif gesture == 'Y':  # Close curtains
        GPIO.output(CURTAIN_PIN, GPIO.LOW)
        print("Curtains CLOSED")

def real_time_detection():
    """Run real-time ASL detection using webcam and TFLite model"""
    # Load TFLite model
    interpreter = load_tflite_model()
    if interpreter is None:
        return
    
    # Load preprocessing data
    label_encoder, X_min, X_max, class_mapping = load_preprocessing_data()
    if label_encoder is None:
        return
    
    # Initialize webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    
    # Handle camera connection error
    retry_count = 0
    while not cap.isOpened() and retry_count < 5:
        print(f"Failed to open camera. Retrying... ({retry_count+1}/5)")
        time.sleep(1)
        cap = cv2.VideoCapture(0)
        retry_count += 1
        
    if not cap.isOpened():
        print("Error: Could not open webcam after multiple attempts.")
        GPIO.cleanup()
        return
    
    # Set camera properties for better performance on Raspberry Pi
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for Pi
    
    # Variables for tracking detection history and stabilizing predictions
    detection_history = []
    last_prediction = None
    prediction_count = 0
    action_triggered = False
    last_action_time = 0
    cooldown_period = 2.0  # seconds
    
    print("Starting detection. Press 'q' to quit.")
    
    # For tracking FPS
    prev_frame_time = 0
    
    # Start detection loop
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame - check camera connection")
                # Try to reconnect
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("Could not reconnect to camera. Exiting.")
                    break
                continue
            
            # Calculate FPS
            current_frame_time = time.time()
            fps = 1 / (current_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
            prev_frame_time = current_frame_time
                
            # Get frame dimensions
            H, W, _ = frame.shape
                
            # Process the frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.flip(frame_rgb, 1)  # Mirror image for more intuitive interaction
            
            # Make detection - to improve performance on Pi, we can process at a lower resolution
            results = hands.process(frame_rgb)
            
            # Convert back to BGR for display
            frame_display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Check for detection cooldown
            if action_triggered and (current_frame_time - last_action_time < cooldown_period):
                # Show countdown for cooldown period
                remaining = cooldown_period - (current_frame_time - last_action_time)
                cv2.putText(
                    frame_display, 
                    f"Cooldown: {remaining:.1f}s", 
                    (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    ORANGE, 
                    2, 
                    cv2.LINE_AA
                )
                if current_frame_time - last_action_time >= cooldown_period:
                    action_triggered = False
                    
            # Process hand landmarks if detected
            current_prediction = None
            if results.multi_hand_landmarks:
                data_aux = []
                x_coords = []
                y_coords = []
                
                # Get the first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame_display,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=GREEN, thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=BLUE, thickness=2, circle_radius=2)
                )
                
                # Extract coordinates
                for landmark in hand_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_coords.append(x)
                    y_coords.append(y)
                
                # Draw bounding box
                x1 = max(0, int(min(x_coords) * W) - 20)
                y1 = max(0, int(min(y_coords) * H) - 20)
                x2 = min(W, int(max(x_coords) * W) + 20)
                y2 = min(H, int(max(y_coords) * H) + 20)
                
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), GREEN, 2)
                
                # Process input data for TFLite model
                processed_data = process_input_data(data_aux, X_min, X_max)
                
                if processed_data is not None:
                    # Make prediction
                    predicted_class, confidence = predict_with_tflite(interpreter, processed_data)
                    current_prediction = class_mapping[predicted_class]
                    
                    # Update detection history for stabilization
                    detection_history.append(current_prediction)
                    if len(detection_history) > 5:
                        detection_history.pop(0)
                    
                    # Only change prediction if we have consistent results
                    if current_prediction == last_prediction:
                        prediction_count += 1
                    else:
                        prediction_count = 1
                    
                    # Stabilize prediction using history
                    if prediction_count >= 3:
                        # Get the most common prediction in history
                        from collections import Counter
                        stable_prediction = Counter(detection_history).most_common(1)[0][0]
                        
                        # Get action associated with gesture
                        action = ACTION_MAP.get(stable_prediction, "Unknown gesture")
                        
                        # Draw prediction box
                        cv2.rectangle(frame_display, (x1, y1-60), (x2, y1-10), PURPLE, -1)
                        cv2.putText(
                            frame_display, 
                            f"{stable_prediction} - {action}", 
                            (x1+10, y1-25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, 
                            (255, 255, 255), 
                            2, 
                            cv2.LINE_AA
                        )
                        
                        # Add confidence display
                        cv2.putText(
                            frame_display,
                            f"Confidence: {confidence:.2f}",
                            (x1+10, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA
                        )
                        
                        # Trigger action if not in cooldown
                        if not action_triggered and stable_prediction in ACTION_MAP:
                            print(f"Action triggered: {action}")
                            # Control smart home based on the gesture
                            control_smart_home(stable_prediction)
                            action_triggered = True
                            last_action_time = current_frame_time
                    
                    last_prediction = current_prediction
            
            # Display current device states
            light_status = "ON" if GPIO.input(LIGHT_PIN) else "OFF"
            fan_status = "ON" if GPIO.input(FAN_PIN) else "OFF"
            curtain_status = "OPEN" if GPIO.input(CURTAIN_PIN) else "CLOSED"
            
            cv2.putText(
                frame_display, 
                f"Lights: {light_status}", 
                (W-150, H-80), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                YELLOW if GPIO.input(LIGHT_PIN) else RED, 
                2, 
                cv2.LINE_AA
            )
            
            cv2.putText(
                frame_display, 
                f"Fan: {fan_status}", 
                (W-150, H-50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                YELLOW if GPIO.input(FAN_PIN) else RED, 
                2, 
                cv2.LINE_AA
            )
            
            cv2.putText(
                frame_display, 
                f"Curtains: {curtain_status}", 
                (W-150, H-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                YELLOW if GPIO.input(CURTAIN_PIN) else RED, 
                2, 
                cv2.LINE_AA
            )
            
            # Display title
            cv2.putText(
                frame_display, 
                "ASL Smart Home Control", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, 
                BLUE, 
                2, 
                cv2.LINE_AA
            )
            
            # Display FPS
            cv2.putText(
                frame_display,
                f"FPS: {fps:.1f}",
                (W-120, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                GREEN,
                2,
                cv2.LINE_AA
            )
            
            # Show instructions
            cv2.putText(
                frame_display, 
                "Press 'q' to quit", 
                (10, H-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                RED, 
                2, 
                cv2.LINE_AA
            )
            
            # Display the frame
            cv2.imshow('ASL Smart Home Control', frame_display)
            
            # Break the loop when 'q' is pressed
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
                
        # Release resources
        print("Releasing resources...")
        cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()  # Clean up GPIO
        print("Detection stopped.")

if __name__ == "__main__":
    try:
        real_time_detection()
    except KeyboardInterrupt:
        print("Program stopped by user")
        GPIO.cleanup()  # Ensure GPIO is cleaned up on keyboard interrupt
    except Exception as e:
        print(f"An error occurred: {e}")
        GPIO.cleanup()  # Ensure GPIO is cleaned up on any exception