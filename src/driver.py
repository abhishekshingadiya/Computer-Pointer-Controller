from argparse import ArgumentParser
import numpy as np
import logging
from input_feeder import InputFeeder
import constants
import os
from face_detection import Face_Model
from facial_landmarks_detection import Landmark_Model
from gaze_estimation import Gaze_Estimation_Model
from head_pose_estimation import Head_Pose_Model
from mouse_controller import MouseController
import cv2
import imutils
import math


# import line_profiler
# profile=line_profiler.LineProfiler()
# import atexit
# atexit.register(profile.print_stats)


def imshow(windowname, frame, width=None):
    if width == None:
        width = 400

    frame = imutils.resize(frame, width=width)
    cv2.imshow(windowname, frame)


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-f", "--face", required=True, type=str,
                        help="Path to .xml file of Face Detection model.")
    parser.add_argument("-l", "--landmarks", required=True, type=str,
                        help="Path to .xml file of Facial Landmark Detection model.")
    parser.add_argument("-hp", "--headpose", required=True, type=str,
                        help="Path to .xml file of Head Pose Estimation model.")
    parser.add_argument("-ge", "--gazeestimation", required=True, type=str,
                        help="Path to .xml file of Gaze Estimation model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file or enter cam for webcam")
    parser.add_argument("-it", "--input_type", required=True, type=str,
                        help="Provide the source of video frames." + constants.VIDEO + " " + constants.WEBCAM + " | " + constants.IP_CAMERA + " | " + constants.IMAGE)
    parser.add_argument("-debug", "--debug", required=False, type=str, nargs='+',
                        default=[],
                        help="To debug each model's output visually, type the model name with comma seperated after --debug")
    parser.add_argument("-ld", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="linker libraries if have any")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Provide the target device: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable.")

    return parser


# @profile
def main(args):
    logger = logging.getLogger()

    feeder = None
    if args.input_type == constants.VIDEO or args.input_type == constants.IMAGE:
        extension = str(args.input).split('.')[1]
        # if not extension.lower() in constants.ALLOWED_EXTENSIONS:
        #     logger.error('Please provide supported extension.' + str(constants.ALLOWED_EXTENSIONS))
        #     exit(1)

        # if not os.path.isfile(args.input):
        #     logger.error("Unable to find specified video/image file")
        #     exit(1)

        feeder = InputFeeder(args.input_type, args.input)
    elif args.input_type == constants.IP_CAMERA:
        if not str(args.input).startswith('http://'):
            logger.error('Please provide ip of server with http://')
            exit(1)

        feeder = InputFeeder(args.input_type, args.input)
    elif args.input_type == constants.WEBCAM:
        feeder = InputFeeder(args.input_type)

    mc = MouseController("medium", "fast")

    feeder.load_data()

    face_model = Face_Model(args.face, args.device, args.cpu_extension)
    face_model.check_model()

    landmark_model = Landmark_Model(args.landmarks, args.device, args.cpu_extension)
    landmark_model.check_model()

    gaze_model = Gaze_Estimation_Model(args.gazeestimation, args.device, args.cpu_extension)
    gaze_model.check_model()

    head_model = Head_Pose_Model(args.headpose, args.device, args.cpu_extension)
    head_model.check_model()

    face_model.load_model()
    logger.info("Face Detection Model Loaded...")
    landmark_model.load_model()
    logger.info("Landmark Detection Model Loaded...")
    gaze_model.load_model()
    logger.info("Gaze Estimation Model Loaded...")
    head_model.load_model()
    logger.info("Head Pose Detection Model Loaded...")
    print('Loaded')

    w = int(feeder.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(feeder.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(feeder.cap.get(cv2.CAP_PROP_FPS))
    fourcc = int(feeder.cap.get(cv2.CAP_PROP_FOURCC))
    out_video = cv2.VideoWriter(os.path.join('output_video.mp4'), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                (w, h), True)

    preview_flags = []
    # ['fl', 'fh', 'fg', 'ff']
    try:
        frame_count = 0
        for ret, frame in feeder.next_batch():
            if not ret:
                break
            frame_count += 1
            key = cv2.waitKey(1)

            try:
                cropped_image, face_cords = face_model.predict(frame)

                if type(cropped_image) == int:
                    print("Unable to detect the face")
                    if key == 27:
                        break
                    continue

                (lefteye_x, lefteye_y), (
                    righteye_x, righteye_y), eye_cords, left_eye, right_eye = landmark_model.predict(cropped_image,
                                                                                                     eye_surrounding_area=15)
                pose_output = head_model.predict(cropped_image)
                mouse_cord, gaze_vector = gaze_model.predict(left_eye, right_eye, pose_output)
            except Exception as e:
                print(str(e) + " for frame " + str(frame_count))
                continue

            image = cv2.resize(frame, (w, h))
            if not len(preview_flags) == 0:
                preview_frame = frame.copy()
                const = 10
                if 'ff' in preview_flags:
                    if len(preview_flags) != 1:
                        preview_frame = cropped_image
                        cv2.rectangle(frame, (face_cords[0], face_cords[1]), (face_cords[2], face_cords[3]),
                                      (255, 0, 0), 3)

                if 'fl' in preview_flags:
                    cv2.rectangle(cropped_image, (eye_cords[0][0] - const, eye_cords[0][1] - const),
                                  (eye_cords[0][2] + const, eye_cords[0][3] + const),
                                  (0, 255, 0), 2)
                    cv2.rectangle(cropped_image, (eye_cords[1][0] - const, eye_cords[1][1] - const),
                                  (eye_cords[1][2] + const, eye_cords[1][3] + const),
                                  (0, 255, 0), 2)

                if 'fh' in preview_flags:
                    cv2.putText(
                        frame,
                        "Pose Angles: yaw= {:.2f} , pitch= {:.2f} , roll= {:.2f}".format(
                            pose_output[0], pose_output[1], pose_output[2]),
                        (20, 40),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1, (255, 0, 255), 2)

                if 'fg' in preview_flags:
                    x, y, w = int(gaze_vector[0] * 12), int(gaze_vector[1] * 12), 160
                    le = cv2.line(left_eye.copy(), (x - w, y - w), (x + w, y + w), (255, 0, 255), 2)
                    cv2.arrowedLine(le, (x - w, y + w), (x + w, y - w), (255, 0, 255), 2)
                    re = cv2.line(right_eye.copy(), (x - w, y - w), (x + w, y + w), (255, 0, 255), 2)
                    cv2.arrowedLine(re, (x - w, y + w), (x + w, y - w), (255, 0, 255), 2)
                    preview_frame[eye_cords[0][1]:eye_cords[0][3], eye_cords[0][0]:eye_cords[0][2]] = le
                    preview_frame[eye_cords[1][1]:eye_cords[1][3], eye_cords[1][0]:eye_cords[1][2]] = re
                image = np.hstack((cv2.resize(frame, (500, 500)), cv2.resize(preview_frame, (500, 500))))

            cv2.imshow('preview', image)
            out_video.write(frame)

            if frame_count % 20 == 0:
                mc.move(mouse_cord[0], mouse_cord[1])

            if key == 27:
                break

        logger.info('Video stream ended')
        cv2.destroyAllWindows()
        feeder.close()

    except Exception as err:
        logger.error(err)
        cv2.destroyAllWindows()
        feeder.close()


if __name__ == '__main__':
    arg = '-f ../models/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml -l ../models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml -hp ../models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -ge ../models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml -i ../bin/demo.mp4 -it video -d CPU -debug headpose gaze face'.split(
        ' ')
    args = build_argparser().parse_args(arg)
    # args = build_argparser().parse_args()

    main(args)
