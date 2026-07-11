
from safe_exam.detectors.head_pose.head_pose_config import HeadPoseConfig
from safe_exam.detectors.head_pose.head_pose_detector import create_face_mesh, process_head_pose_frame
from safe_exam.capture.capture_loop import capture_frames
from safe_exam.capture.config import CaptureConfig
import cv2


config = HeadPoseConfig()
face_mesh = create_face_mesh(config)

def main() -> None:
    config = HeadPoseConfig()
    capture_config = CaptureConfig()
    face_mesh = create_face_mesh(config)

    try:
        for frame in capture_frames(capture_config):
            display, direction, angles = process_head_pose_frame(
                frame,
                face_mesh,
                config,
            )

            cv2.imshow("Head Pose", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        face_mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()