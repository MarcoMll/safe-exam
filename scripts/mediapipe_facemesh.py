import mediapipe as mp
import cv2
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

#opening webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # we would use .imread reads an image from a file into a NumPy array.
    # it depends on the context, .read reads a frames from the webcam
    successful,image = cap.read()

    if not successful:
        continue

    # flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB. (cv2 uses BGR format, while mediapipe uses RGB format)
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # apparently this improves perfomance
    image.flags.writeable = False # makes it so we can only read the image when passing it

    # get the results
    # gives us the face landmarks
    results = face_mesh.process(image)

    # improve performace by making the image writeable again
    image.flags.writeable = True

    # convert the image back to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # storing image height, width and channels
    img_h, img_w, img_c = image.shape


    # detecting if there are any face landmarks
    if results.multi_face_landmarks:
        # going through all the detected face landmarks
        for face_landmarks in results.multi_face_landmarks:
                # mapping the 3D and 2D points of the face mesh
            face_3d = []
            face_2d = []
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 1 or idx == 33 or idx == 61 or idx == 199 or idx == 263 or idx == 291:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    
                    # getting 2d coordinates
                    face_2d.append([x, y])
                    
                    # getting 3d coordinates
                    face_3d.append([x, y, lm.z])
                
                # converting the lists to numpy arrays
                if len(face_2d) >= 6:
                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    # the camera matrix
                    focal_length = 1 * img_w
                    
                    cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                        [0, focal_length, img_h / 2],
                                        [0, 0, 1]])
                    
                    # distortion parameters (no distortion here)
                    # not ideal for later but for a prototype this is fine
                    # it assumes there is no distortion in the camera lens
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    # Solve PnP which is function:
                    # used to estimate the 3D pose of an object or camera 
                    # by computing rotation and translation vectors
                    ok, roc_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                    if ok:
                        # convert roc_vec (rotation vector) to rotation matrix (we will not use jacobian matrix (jac))
                        rmat, jac = cv2.Rodrigues(roc_vec)

                        # get the angles for all the axes
                        # using cv2 RQDecomp3x3 function to decompose the rotation matrix into angles
                        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                        # get the y rotation degree
                        x = angles[0] * 360
                        y = angles[1] * 360
                        z = angles[2] * 360

                        # now we can see if the head is tilting
                        if y < -10:
                            text = "Looking Left"
                        elif y > 10:
                            text = "Looking Right"
                        elif x < -10:
                            text = "Looking Down"
                        elif x > 10:
                            text = "Looking Up"
                        else:
                            text = "Forward"

                        # display the nose direction
                        nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, roc_vec, trans_vec, cam_matrix, dist_matrix)

                        p1 = (int(nose_2d[0]), int(nose_2d[1]))
                        p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                        cv2.line(image, p1, p2, (255, 0, 0), 3)

                        # adding text to the image
                        # the cordinates and the text of the direction in which we are looking
                        cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
            # drawing the face landmarks on the image
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
            
    cv2.imshow('Head Pose Estimation', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()