from flask import Blueprint, Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize Flask app
biceps_bp = Blueprint('biceps', __name__, template_folder='../templates')

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Vector formation
    ba = a - b
    bc = c - b
    
    # Cosine law for angle
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    # Convert to degrees
    angle = np.degrees(angle)
    
    return angle

# Initialize variables for bicep curls
bicep_counter = 0
bicep_stage = None
feedback_text = "Position yourself in frame"
last_feedback_time = time.time()  # Initialize with current time
feedback_duration = 2  # seconds to display each feedback
confidence_threshold = 0.5  # Minimum confidence for detection - reduced threshold
smoothing_window = []  # For angle smoothing
max_smoothing_values = 5
last_angle = 0

# Form check parameters
min_curl_angle = 40
max_curl_angle = 160
ideal_curl_range = (30, 150)  # Ideal range for a good bicep curl

def live_video_feed():
    global bicep_counter, bicep_stage, feedback_text, last_feedback_time, smoothing_window, last_angle
    
    cap = cv2.VideoCapture(0)
    
    # Set higher resolution if camera supports it
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with mp_pose.Pose(
        min_detection_confidence=0.5,  # Reduced to improve detection rate
        min_tracking_confidence=0.5,
        model_complexity=1  # Use a more accurate model (0, 1, or 2)
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Get the current time for feedback timing
            current_time = time.time()
            
            # Resize for faster processing while maintaining aspect ratio
            frame_height, frame_width = frame.shape[:2]
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Convert back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                if results.pose_landmarks is None:
                    raise Exception("No pose detected")
                
                landmarks = results.pose_landmarks.landmark
                
                # Check if landmarks have sufficient visibility
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                
                # Check if at least one arm is visible enough
                left_arm_visible = (left_shoulder.visibility > confidence_threshold and 
                                  left_elbow.visibility > confidence_threshold and 
                                  left_wrist.visibility > confidence_threshold)
                
                right_arm_visible = (right_shoulder.visibility > confidence_threshold and 
                                   right_elbow.visibility > confidence_threshold and 
                                   right_wrist.visibility > confidence_threshold)
                
                # Choose the arm that's more visible
                if left_arm_visible and (not right_arm_visible or left_elbow.visibility > right_elbow.visibility):
                    # Use left arm
                    shoulder = [left_shoulder.x, left_shoulder.y]
                    elbow = [left_elbow.x, left_elbow.y]
                    wrist = [left_wrist.x, left_wrist.y]
                    arm_text = "Left Arm"
                elif right_arm_visible:
                    # Use right arm
                    shoulder = [right_shoulder.x, right_shoulder.y]
                    elbow = [right_elbow.x, right_elbow.y]
                    wrist = [right_wrist.x, right_wrist.y]
                    arm_text = "Right Arm"
                else:
                    # No arms visible enough
                    if current_time - last_feedback_time > feedback_duration:
                        feedback_text = "Move closer or show your arms clearly"
                        last_feedback_time = current_time
                    raise Exception("Arms not clearly visible")

                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)
                
                # Smooth the angle with a moving average filter
                smoothing_window.append(angle)
                if len(smoothing_window) > max_smoothing_values:
                    smoothing_window.pop(0)
                
                smoothed_angle = sum(smoothing_window) / len(smoothing_window)
                angle_change = abs(smoothed_angle - last_angle)
                last_angle = smoothed_angle
                
                # Convert coordinates to pixel position for drawing
                shoulder_px = (int(shoulder[0] * frame_width), int(shoulder[1] * frame_height))
                elbow_px = (int(elbow[0] * frame_width), int(elbow[1] * frame_height))
                wrist_px = (int(wrist[0] * frame_width), int(wrist[1] * frame_height))
                
                # Draw arm angle lines and points
                cv2.line(frame, shoulder_px, elbow_px, (255, 255, 0), 3)
                cv2.line(frame, elbow_px, wrist_px, (255, 255, 0), 3)
                cv2.circle(frame, shoulder_px, 8, (0, 0, 255), -1)
                cv2.circle(frame, elbow_px, 8, (0, 0, 255), -1)
                cv2.circle(frame, wrist_px, 8, (0, 0, 255), -1)
                
                # Put angle text near elbow
                cv2.putText(frame, f"Angle: {smoothed_angle:.1f}", 
                            (elbow_px[0]-50, elbow_px[1]-20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Put arm text
                cv2.putText(frame, arm_text, 
                            (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Rep counting logic with improved accuracy
                
                # Down position - arm straight
                if smoothed_angle > 140:  
                    if bicep_stage != "down":
                        bicep_stage = "down"
                        feedback_text = "Now curl up slowly"
                        last_feedback_time = current_time
                    
                    # Check if arm is too straight
                    if smoothed_angle > max_curl_angle and current_time - last_feedback_time > feedback_duration:
                        feedback_text = "Don't lock your elbow completely"
                        last_feedback_time = current_time
                
                # Up position - arm bent
                elif smoothed_angle < 50 and bicep_stage == "down":
                    bicep_stage = "up"
                    bicep_counter += 1
                    feedback_text = f"Rep {bicep_counter} complete! Lower slowly"
                    last_feedback_time = current_time
                
                # In between - check for form issues
                elif bicep_stage == "down" and 80 < smoothed_angle < 110 and current_time - last_feedback_time > feedback_duration:
                    if angle_change < 2:  # Detect if movement is stalled
                        feedback_text = "Keep moving - don't pause halfway"
                        last_feedback_time = current_time
                
                # Check if person is moving too quickly
                elif angle_change > 15 and current_time - last_feedback_time > feedback_duration:
                    feedback_text = "Move more slowly for better results"
                    last_feedback_time = current_time

            except Exception as e:
                # Make sure current_time is defined
                current_time = time.time()
                
                # Reset if person moves out of frame
                if current_time - last_feedback_time > feedback_duration:
                    feedback_text = "Position yourself in frame"
                    last_feedback_time = current_time

            # Draw feedback box at the bottom
            cv2.rectangle(frame, (0, frame_height-70), (frame_width, frame_height), (0, 0, 0), -1)
            cv2.putText(frame, f'Bicep Curls: {bicep_counter}', (10, frame_height-40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, feedback_text, (10, frame_height-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Draw pose landmarks if available
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

            # Encode the frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()
        cv2.destroyAllWindows()

# Function to stream feedback video with text at the bottom
def feedback_display():
    global bicep_counter, feedback_text
    video_path = r'C:\Users\venkateshwaran\OneDrive\Documents\final (march ) AI (SRH)\final (march ) AI (SRH) after git\Ai\static\videos\WhatsApp Video 2024-12-09 at 17.14.07_16f08022.mp4'
    cap = cv2.VideoCapture(video_path)

    # Check if video file exists and is opened successfully
    if not cap.isOpened():
        # Create a blank frame with error message if video can't be loaded
        while True:
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Video file not found or cannot be opened", 
                        (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add counter and feedback
            cv2.rectangle(frame, (0, 320), (640, 360), (0, 0, 0), -1)
            cv2.putText(frame, f'Bicep Curls: {bicep_counter}', (10, 345), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, feedback_text, (220, 345), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            feedback_frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + feedback_frame + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
    else:
        # Set video frame rate (fps)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default to 30fps if unable to get proper fps

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart the video once it finishes
                continue

            # Resize the video frame to a smaller size
            frame = cv2.resize(frame, (640, 360))

            # Add feedback text to the bottom of the frame
            cv2.rectangle(frame, (0, 320), (640, 360), (0, 0, 0), -1)  # Black background for text
            cv2.putText(frame, f'Bicep Curls: {bicep_counter}', (10, 345), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, feedback_text, (220, 345), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Encode the frame
            ret, buffer = cv2.imencode('.jpg', frame)
            feedback_frame = buffer.tobytes()

            # Yield the frame to Flask
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + feedback_frame + b'\r\n')

            # Wait to control the frame rate
            cv2.waitKey(int(1000 / fps))  # Wait based on the original frame rate

    cap.release()

# Routes
@biceps_bp.route('/')
def index():
    return render_template('bicep.html')

@biceps_bp.route('/video_feed')
def video_feed():
    return Response(live_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@biceps_bp.route('/feedback')
def feedback():
    return Response(feedback_display(), mimetype='multipart/x-mixed-replace; boundary=frame')