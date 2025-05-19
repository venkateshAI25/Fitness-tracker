from flask import Blueprint, Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize Flask app with correct template path
squats_bp = Blueprint('squats', __name__, template_folder='templates')

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

# Function to calculate angle
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

# Initialize variables for squats
squat_counter = 0
squat_stage = "up"  # Start with assumption that person is standing
feedback_text = "Position yourself in frame"
last_feedback_time = time.time()
feedback_duration = 2  # seconds to display each feedback
confidence_threshold = 0.5  # Minimum confidence for detection
knee_angle_smoothing = []
hip_angle_smoothing = []
max_smoothing_values = 5
last_knee_angle = 0
last_hip_angle = 0

# Enhanced rep counting variables
squat_positions = {
    "standing": {"min_angle": 150, "max_angle": 180},
    "quarter": {"min_angle": 120, "max_angle": 150},
    "half": {"min_angle": 100, "max_angle": 120},
    "deep": {"min_angle": 70, "max_angle": 100}
}
squat_depth_history = []
max_depth_history = 5
rep_state = "standing"  # More detailed state tracking
min_rep_duration = 0.7  # Minimum time (seconds) a complete rep should take
last_rep_time = time.time()
rep_threshold_angles = {"down": 120, "up": 140}  # Thresholds for detecting up/down transitions
required_depth = "half"  # Minimum depth required for a valid rep
depth_consistency_threshold = 15  # Maximum allowed variation in depth (degrees)
false_rep_timeout = 0.3  # Seconds to wait before counting another rep to prevent false counts
valid_rep_depths = []  # Store depths of valid reps for analysis

# Form check parameters
min_knee_angle = 70    # Minimum knee angle for a good squat
max_knee_angle = 170   # Max knee angle (standing)
min_hip_angle = 70     # Minimum hip angle for a good squat
max_hip_angle = 170    # Max hip angle (standing)
ideal_squat_range = (70, 100)  # Ideal range for a good squat (knee angle)

def live_video_feed():
    global squat_counter, squat_stage, feedback_text, last_feedback_time
    global knee_angle_smoothing, hip_angle_smoothing, last_knee_angle, last_hip_angle
    global rep_state, last_rep_time, squat_depth_history, valid_rep_depths
    
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

            # Get current time for feedback timing
            current_time = time.time()
            
            # Get frame dimensions
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
                
                # Check if the person is facing sideways (profile view)
                # For squat detection, a side view is best for angle calculations
                
                # Get hip, knee and ankle coordinates
                # We'll try both left and right side and use the more visible one
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                
                # Check visibility
                left_side_visible = (left_hip.visibility > confidence_threshold and
                                   left_knee.visibility > confidence_threshold and
                                   left_ankle.visibility > confidence_threshold and
                                   left_shoulder.visibility > confidence_threshold)
                
                right_side_visible = (right_hip.visibility > confidence_threshold and
                                    right_knee.visibility > confidence_threshold and
                                    right_ankle.visibility > confidence_threshold and
                                    right_shoulder.visibility > confidence_threshold)
                
                # Choose the side that's more visible
                if left_side_visible and (not right_side_visible or left_knee.visibility > right_knee.visibility):
                    # Use left side
                    hip = [left_hip.x, left_hip.y]
                    knee = [left_knee.x, left_knee.y]
                    ankle = [left_ankle.x, left_ankle.y]
                    shoulder = [left_shoulder.x, left_shoulder.y]
                    side_text = "Left Side"
                elif right_side_visible:
                    # Use right side
                    hip = [right_hip.x, right_hip.y]
                    knee = [right_knee.x, right_knee.y]
                    ankle = [right_ankle.x, right_ankle.y]
                    shoulder = [right_shoulder.x, right_shoulder.y]
                    side_text = "Right Side"
                else:
                    # No sides visible enough
                    if current_time - last_feedback_time > feedback_duration:
                        feedback_text = "Please show your full body from the side"
                        last_feedback_time = current_time
                    raise Exception("Body not clearly visible")
                
                # Calculate knee angle (between hip, knee, and ankle)
                knee_angle = calculate_angle(hip, knee, ankle)
                
                # Calculate hip angle (between shoulder, hip, and knee)
                hip_angle = calculate_angle(shoulder, hip, knee)
                
                # Smooth the angles with moving average filters
                knee_angle_smoothing.append(knee_angle)
                hip_angle_smoothing.append(hip_angle)
                
                if len(knee_angle_smoothing) > max_smoothing_values:
                    knee_angle_smoothing.pop(0)
                if len(hip_angle_smoothing) > max_smoothing_values:
                    hip_angle_smoothing.pop(0)
                
                smoothed_knee_angle = sum(knee_angle_smoothing) / len(knee_angle_smoothing)
                smoothed_hip_angle = sum(hip_angle_smoothing) / len(hip_angle_smoothing)
                
                knee_angle_change = abs(smoothed_knee_angle - last_knee_angle)
                hip_angle_change = abs(smoothed_hip_angle - last_hip_angle)
                
                last_knee_angle = smoothed_knee_angle
                last_hip_angle = smoothed_hip_angle
                
                # Convert coordinates to pixel positions for drawing
                hip_px = (int(hip[0] * frame_width), int(hip[1] * frame_height))
                knee_px = (int(knee[0] * frame_width), int(knee[1] * frame_height))
                ankle_px = (int(ankle[0] * frame_width), int(ankle[1] * frame_height))
                shoulder_px = (int(shoulder[0] * frame_width), int(shoulder[1] * frame_height))
                
                # Draw leg lines and points
                cv2.line(frame, hip_px, knee_px, (255, 255, 0), 3)
                cv2.line(frame, knee_px, ankle_px, (255, 255, 0), 3)
                cv2.line(frame, shoulder_px, hip_px, (255, 255, 0), 3)
                
                cv2.circle(frame, hip_px, 8, (0, 0, 255), -1)
                cv2.circle(frame, knee_px, 8, (0, 0, 255), -1)
                cv2.circle(frame, ankle_px, 8, (0, 0, 255), -1)
                cv2.circle(frame, shoulder_px, 8, (0, 0, 255), -1)
                
                # Put angle texts
                cv2.putText(frame, f"Knee: {smoothed_knee_angle:.1f}°", 
                            (knee_px[0]-60, knee_px[1]-20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.putText(frame, f"Hip: {smoothed_hip_angle:.1f}°", 
                            (hip_px[0]-60, hip_px[1]-20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Put side text
                cv2.putText(frame, side_text, 
                            (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Determine current position based on knee angle
                current_position = "unknown"
                for position, angle_range in squat_positions.items():
                    if angle_range["min_angle"] <= smoothed_knee_angle <= angle_range["max_angle"]:
                        current_position = position
                        break

                # Display current position
                position_colors = {
                    "standing": (0, 255, 0),    # Green
                    "quarter": (0, 255, 255),   # Yellow
                    "half": (0, 165, 255),      # Orange
                    "deep": (0, 0, 255)         # Red
                }
                position_color = position_colors.get(current_position, (255, 255, 255))  # Default white
                cv2.putText(frame, f'Position: {current_position}', (frame_width-220, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, position_color, 2)
                
                # Improved state machine for rep counting
                if rep_state == "standing" and smoothed_knee_angle < rep_threshold_angles["down"]:
                    # Transition from standing to squatting
                    rep_state = "descending"
                    feedback_text = "Going down - keep your back straight"
                    last_feedback_time = current_time
                        
                elif rep_state == "descending" and smoothed_knee_angle < squat_positions[required_depth]["max_angle"]:
                    # Reached required depth, record the actual depth
                    rep_state = "bottom"
                    current_depth = smoothed_knee_angle
                    squat_depth_history.append(current_depth)
                    if len(squat_depth_history) > max_depth_history:
                        squat_depth_history.pop(0)
                    feedback_text = "Good depth! Now stand up"
                    last_feedback_time = current_time
                        
                elif rep_state == "bottom" and smoothed_knee_angle > rep_threshold_angles["up"]:
                    # Starting to stand up
                    rep_state = "ascending"
                    feedback_text = "Coming up - push through your heels"
                    last_feedback_time = current_time
                        
                elif rep_state == "ascending" and smoothed_knee_angle > squat_positions["standing"]["min_angle"]:
                    # Completed the rep by returning to standing
                    rep_state = "standing"
                    
                    # Calculate time spent on this rep
                    rep_duration = current_time - last_rep_time
                    last_rep_time = current_time
                    
                    # Check for false reps (too quick)
                    if rep_duration > min_rep_duration:
                        # Check if we reached sufficient depth during this rep
                        if len(squat_depth_history) > 0:
                            min_depth_reached = min(squat_depth_history)
                            
                            # Determine if this was a valid rep
                            if min_depth_reached <= squat_positions[required_depth]["max_angle"]:
                                # Valid rep with good depth
                                squat_counter += 1
                                valid_rep_depths.append(min_depth_reached)
                                
                                # Provide depth consistency feedback
                                if len(valid_rep_depths) > 1:
                                    depth_diff = abs(valid_rep_depths[-1] - valid_rep_depths[-2])
                                    if depth_diff > depth_consistency_threshold:
                                        feedback_text = f"Rep {squat_counter} complete! Try to maintain consistent depth"
                                    else:
                                        feedback_text = f"Rep {squat_counter} complete! Good consistency"
                                else:
                                    feedback_text = f"Rep {squat_counter} complete! Good job!"
                            else:
                                # Not deep enough
                                feedback_text = f"Need to go deeper! Aim for {required_depth} squat depth"
                            
                            last_feedback_time = current_time
                            squat_depth_history = []  # Reset depth history for next rep
                    else:
                        # Rep was too quick
                        feedback_text = "Move more slowly for better results"
                        last_feedback_time = current_time
                
                # Add visual state debugging
                state_color = position_colors.get(current_position, (255, 255, 255))
                cv2.putText(frame, f'State: {rep_state}', (frame_width-180, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
                
                # Form check section
                # Check if person is locking their knees too much
                if smoothed_knee_angle > max_knee_angle and current_time - last_feedback_time > feedback_duration:
                    feedback_text = "Don't lock your knees completely"
                    last_feedback_time = current_time
                
                # Check if knees are bending too much
                elif rep_state == "bottom" and smoothed_knee_angle < min_knee_angle and current_time - last_feedback_time > feedback_duration:
                    feedback_text = "Watch your knees! Don't go too low"
                    last_feedback_time = current_time
                
                # Check for pausing
                elif (rep_state == "descending" or rep_state == "ascending") and knee_angle_change < 2 and current_time - last_feedback_time > feedback_duration:
                    feedback_text = "Keep moving - don't pause halfway"
                    last_feedback_time = current_time
                
                # Check if person is moving too quickly
                elif (knee_angle_change > 15 or hip_angle_change > 15) and current_time - last_feedback_time > feedback_duration:
                    feedback_text = "Move more slowly for better results"
                    last_feedback_time = current_time
                
                # Check if hips are not bending enough during descent
                elif rep_state == "descending" and smoothed_hip_angle > 120 and smoothed_knee_angle < 120 and current_time - last_feedback_time > feedback_duration:
                    feedback_text = "Push your hips back more"
                    last_feedback_time = current_time
                
                # Check if knees are going too far forward (knee should not go beyond toes)
                elif rep_state == "bottom" and knee[0] > ankle[0] + 0.05 and current_time - last_feedback_time > feedback_duration:
                    feedback_text = "Knees going too far forward!"
                    last_feedback_time = current_time
                
            except Exception as e:
                # Make sure current_time is defined
                current_time = time.time()
                
                # Reset if person moves out of frame
                if current_time - last_feedback_time > feedback_duration:
                    feedback_text = "Position yourself in frame - show full body"
                    last_feedback_time = current_time
            
            # Draw feedback box at the bottom
            cv2.rectangle(frame, (0, frame_height-70), (frame_width, frame_height), (0, 0, 0), -1)
            cv2.putText(frame, f'Squats: {squat_counter}', (10, frame_height-40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, feedback_text, (10, frame_height-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Draw pose landmarks
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

# Function to stream demo video with text at the bottom
def feedback_display():
    global squat_counter, feedback_text
    
    # Change this to the path of your squat demo video - use the same format as in biceps.py
    video_path = r'C:\Users\venkateshwaran\OneDrive\Documents\final (march ) AI (SRH)\final (march ) AI (SRH) after git\Ai\static\videos\WhatsApp Video 2024-12-06 at 17.32.15_54dca5ab.mp4'
    cap = cv2.VideoCapture(video_path)
    
    # Check if video file exists and is opened successfully
    if not cap.isOpened():
        # Create a blank frame with instruction text if video can't be loaded
        while True:
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            
            # Draw instruction text
            instructions = [
                "HOW TO DO A PROPER SQUAT:",
                "1. Stand with feet shoulder-width apart",
                "2. Keep your chest up and back straight",
                "3. Push your hips back as if sitting in a chair",
                "4. Lower until thighs are parallel to floor",
                "5. Keep knees in line with toes",
                "6. Push through heels to stand back up"
            ]
            
            for i, line in enumerate(instructions):
                y_position = 50 + (i * 30)
                cv2.putText(frame, line, (20, y_position), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Add counter and feedback
            cv2.rectangle(frame, (0, 300), (640, 360), (0, 0, 0), -1)
            cv2.putText(frame, f'Squats: {squat_counter}', (10, 325), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, feedback_text, (10, 350), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            feedback_frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + feedback_frame + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
    else:
        # Set video frame rate
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
            cv2.rectangle(frame, (0, 300), (640, 360), (0, 0, 0), -1)
            cv2.putText(frame, f'Squats: {squat_counter}', (10, 325), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, feedback_text, (10, 350), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Encode the frame
            ret, buffer = cv2.imencode('.jpg', frame)
            feedback_frame = buffer.tobytes()
            
            # Yield the frame to Flask
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + feedback_frame + b'\r\n')
            
            # Wait to control the frame rate
            cv2.waitKey(int(1000 / fps))
    
    cap.release()

# Routes
@squats_bp.route('/')
def index():
    return render_template('squats.html')

@squats_bp.route('/video_feed')
def video_feed():
    return Response(live_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@squats_bp.route('/feedback')
def feedback():
    return Response(feedback_display(), mimetype='multipart/x-mixed-replace; boundary=frame')