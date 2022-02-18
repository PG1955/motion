"""
Motion for Python3. Emulates MotionEye but uses tne libcamera library
This openCV varaiam is for development on Windows 10 whwr no libcamera library exists.
Arducam low light camera.

Version Date        Description
v1.10   13/01/2022  Initial version of motion.
v1.11   04/02/2022  Add post_capture parameter and buffer,
v1.13   14/02/2022  Added support for bgr colors in the ini file.
v1.14   18/02/2022  Added support for a dummy libcamaer file to allow development on windows.

"""
__author__ = "Peter Goodgame"
__name__ = "motion"
__version__ = "v11.4"

import argparse
import libcamera
from Journal import journal
import cv2
import time
import datetime
import numpy as np
import os
import sys
import signal
import configparser
import subprocess
import logging
import random
from systemd.journal import JournaldLogHandler
from MotionMP4 import MotionMP4


class TriggerMotion:
    sig_usr1 = False

    def __init__(self):
        if not os.name == 'nt':
            signal.signal(signal.SIGUSR1, self.trigger_motion)

    def trigger_motion(self, *args):
        self.sig_usr1 = True


class GracefulKiller:
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.kill_now = True


def readConfiguration(signalNumber, frame):
    print('(SIGHUP) reading configuration')
    return


def display_fps(index):
    display_fps.frame_count += 1
    current = time.time()
    if current - display_fps.start >= 1:
        print("fps: {}".format(display_fps.frame_count))
        display_fps.frame_count = 0
        display_fps.start = current


def put_text(pt_frame, pt_text, pt_color):
    position = (30, 100)  # indent and line
    font_scale = 0.75
    # BGR colours. 
    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)
    pea = (86, 255, 86)
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_type = cv2.LINE_AA
    text_size, _ = cv2.getTextSize(pt_text, font, font_scale, thickness)
    line_height = text_size[1] + 5
    x, y0 = position
    for i, line in enumerate(pt_text.split("\n")):
        y = y0 + i * line_height
        cv2.putText(pt_frame,
                    line,
                    (x, y),
                    font,
                    font_scale,
                    pt_color,
                    thickness,
                    line_type)
    return pt_frame


def write_jpg(wj_frame):
    jpg_path = mp4.get_pathname().replace('mp4', 'jpg')
    print('JPEG Path: {}'.format(jpg_path))
    cv2.imwrite(jpg_path, wj_frame)


def run_cmd(rc_cmd):
    subprocess.call(rc_cmd, shell=True, executable='/bin/bash')


def get_logger():
    logger = logging.getLogger(__name__)
    # journald_handler = JournaldLogHandler()
    # journald_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    # logger.addHandler(journald_handler)
    logger.setLevel(logging.DEBUG)
    return logger


# def start_camera(sc_width, sc_height, sc_framerate):
# log.info('Starting camera...')
# return False, cv2.VideoCapture(0)


def average_movement(am_frame, am_average):
    # Convert the image to grayscale & blur the result
    am_gray = cv2.cvtColor(am_frame, cv2.COLOR_BGR2GRAY)
    am_gray = cv2.GaussianBlur(am_gray, (21, 21), 0)

    # If the average has not yet beet calculated
    # initialise the variables and read the next frame.
    if am_average is None:
        am_average = am_gray.copy().astype("float")
        am_base_frame = am_frame
        # cam.returnFrameBuffer(data)
        return [], []

    # Accumulate the weighted average between the current frame and
    # previous frames, then compute the difference between the current
    # frame and running average
    cv2.accumulateWeighted(am_gray, am_average, 0.05)
    am_framedelta = cv2.absdiff(am_gray, cv2.convertScaleAbs(am_average))

    # Convert the difference into binary & dilate the result to fill in small holes
    am_thresh = cv2.threshold(am_framedelta, 25, 255, cv2.THRESH_BINARY)[1]
    am_thresh = cv2.dilate(am_thresh, None, iterations=2)

    # Find contours or continuous white blobs in the image
    am_contours, am_hierarchy = cv2.findContours(am_thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return am_contours, am_hierarchy


def draw_box(db_frame, db_label, db_contour, db_color):
    # draw a bounding box/rectangle around the largest contour
    x, y, w, h = cv2.boundingRect(db_contour)
    cv2.rectangle(db_frame, (x, y), (x + w, y + h), db_color, 2)
    # cv2.rectangle(db_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv_area = cv2.contourArea(db_contour)
    cv2.putText(db_frame, db_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  db_color, 2)
    return db_frame


def write_timestamp(wt_frame):
    # Write data and time on the video.
    wt_now = datetime.datetime.now()
    wt_text = wt_now.strftime("%Y/%m/%d %H:%M")
    wt_font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(wt_frame, wt_text,
                (450, 460),
                wt_font, 1,
                (255, 255, 255),
                4, cv2.LINE_8)
    return wt_frame


def get_parameter(gp_parser, gp_name, gp_default):
    try:
        gp_ret = gp_parser.get('Motion', gp_name)
        print('{}: {}'.format(gp_name, gp_ret))
    except:
        gp_ret = gp_default
    return gp_ret


def get_bgr(gb_str):
    red, green, blue = [int(c) for c in gb_str.split(',')]
    return (blue,green,red)
    

def next_index(_index, _buffer_size):
    _index += 1
    if _index >= _buffer_size:
        _index = 0
    return _index


if __name__ == "motion":
    software_version = __version__

    # Check for arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug',
                        action='store_true',
                        help='Debug enabled'
                        )
    args = parser.parse_args()
    # get an instance of the logger object this module will use
    log = get_logger()

    if args.debug:
        log.info('Debug os on')

    log.info('Software Version {}'.format(__version__))
    log.info('PID: {}'.format(os.getpid()))
    print('PID: {}'.format(os.getpid()))

    # Enable signals.
    killer = GracefulKiller()
    motion = TriggerMotion()

    # Read INI file.
    parser = configparser.ConfigParser()
    parser.read('motion.ini')

    framerate = int(get_parameter(parser, 'framerate', 30))
    width = int(get_parameter(parser, 'width', '640'))
    height = int(get_parameter(parser, 'height', '480'))
    sensitivity = int(get_parameter(parser, 'sensitivity', 0))
    stabilise = int(get_parameter(parser, 'stabilise', '10'))
    exposure = int(get_parameter(parser, 'exposure', '0'))
    rotate = int(get_parameter(parser, 'rotate', '0'))
    box = get_parameter(parser, 'box', 'OFF')
    box_bgr = get_bgr(get_parameter(parser, 'box_bgr', '255,255,255'))
    command = get_parameter(parser, 'command', 'None')
    pre_frames = int(get_parameter(parser, 'pre_frames', '0'))
    post_frames = int(get_parameter(parser, 'post_frames', '0'))
    grace_frames = int(get_parameter(parser, 'grace_frames', '0'))
    output_dir = get_parameter(parser, 'output_dir', '.')
    display = bool(get_parameter(parser, 'display', False))
    statistics = bool(get_parameter(parser, 'statistics', False))
    statistics_bgr = get_bgr(get_parameter(parser, 'statistics_bgr', '255,255,255'))

    if args.debug:
        log.info('BOX set to: {}'.format(box))

    # Read the version ini file.
    parser = configparser.ConfigParser()
    parser.read('version.ini')
    version = int(parser.get('MP4', 'version'))

    # ===============================
    # 1) Instantiate the libcamera class
    # ===============================
    cam = libcamera.libcamera()

    """
    2) Initialize the camera
    :param1 width: Set image width
    :param2 height: Set image height
    :param3 pixelFormat: Set image format
    :param4 buffercount: Set the number of buffers
    :param5 rotation: Set the image rotation angle
    :returns ret: Whether the camera is initialized successfully
    """
    ret = cam.initCamera(width, height, libcamera.PixelFormat.RGB888, buffercount=4, rotation=0)

    if not ret:
        # ============================
        # 3) Start Camera
        # ============================
        ret = cam.startCamera()

        frame_time = 1000000 // framerate
        cam.set(libcamera.FrameDurationLimits, [frame_time, frame_time])
        # ret, cap = start_camera(width, height, framerate)

        # Initialise Variables
        size = (width, height)
        average = None
        writer = None
        stabilisation_cnt = 0
        frames_required = 0  # Frames requested for this mp4 file
        frames_written = 0  # Frames written to this mp4 file.
        log.info('Initialise MP4 output')

        # Initlalise video buffer.
        index = 0
        buffer = np.zeros((pre_frames, height, width, 3), np.dtype('uint8'))
        buffered_frame = np.zeros((1, height, width, 3), np.dtype('uint8'))
        jpg_frame = np.zeros((1, height, width, 3), np.dtype('uint8'))

        mp4 = MotionMP4(output_dir, size, version)

        log.info('Camera started')

        if not int(exposure) == 0:
            log.info('Set exposure to {}'.format(exposure))
            # cam.set(libcamera.ExposureTime, int(exposure))

        log.info('PID: {}'.format(os.getpid()))
        # Read images and process them.
        jpg_contour = 0
        movement_flag = False
        frames_required = 0
        contour = (0, 0, 0, 0)
        stablised = False

        # Main process loop.
        while not killer.kill_now:
            # Get a frame.
            # ret, frame = cap.read()
            """
            Read image information

            :returns ret: Whether the image is successfully read
            :returns data: Image data information
            """
            ret, data = cam.readFrame()

            if not ret:
                continue

            # Get image data
            # At present, only RGB888, BGR888, XRGB8888 can be displayed directly, no conversion
            frame = data.imageData

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

            # Convert the image to grayscale & blur the result
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            # If the average has not yet beet calculated
            # initialise the variables and read the next frame.
            if average is None:
                average = gray.copy().astype("float")
                base_frame = frame
                continue

            # Accumulate the weighted average between the current frame and
            # previous frames, then compute the difference between the current
            # frame and running average
            cv2.accumulateWeighted(gray, average, 0.05)
            frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(average))

            # Convert the difference into binary & dilate the result to fill in small holes
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)

            # Find contours or continuous white blobs in the image
            contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Stabilise the camera
            if not stablised:
                stabilisation_cnt += 1
                if stabilisation_cnt == 1:
                    log.info('Initialisation stabilising')
                if stabilisation_cnt == stabilise - 1:
                    log.info('Initialisation stabilisation completed.')
                if stabilisation_cnt < stabilise:
                    cam.returnFrameBuffer(data)
                    continue
            stablised = True

            # Add timestamp.
            frame = write_timestamp(frame)

            if display:
                cv2.imshow('Live Data', frame)

            # Display box.
            if len(contours) > 0:
                areas = [cv2.contourArea(c) for c in contours]
                max_index = np.argmax(areas)
                contour = contours[max_index]
                if not box == 'OFF':
                    if not contour == (0, 0, 0, 0):
                        box_text = box.replace('<value>', str(len(contours)))
                        draw_box(frame, box_text, contour, box_bgr)

            # ==========================================================`
            # save ts frame to the buffer and put the ldest frame in bf.
            # ==========================================================`
            buffer[index] = frame
            index = next_index(index, pre_frames)
            buffered_frame = buffer[index]

            if args.debug:
                log.info('contour:{}'.format(contour))
                log.info('max_index:{}'.format(max_index))

            # If SIGUSR1 trigger a mp4 manually.
            if motion.sig_usr1:
                log.info('Manual SIGUSR1 detected.')
                movement_flag = True
                frames_required = pre_frames + post_frames
                motion.sig_usr1 = False
                jpg_frame = buffered_frame

            # Check if there is any movement, if there is set the movement flag.
            if len(contours) > sensitivity:
                log.info('Motion detected. contour length:{}'.format(str(len(contours))))
                if not movement_flag:
                    movement_flag = True
                    frames_required = post_frames + pre_frames
                else:
                    if frames_written < pre_frames:
                        frames_required = (pre_frames - frames_written) + post_frames
                    else:
                        frames_required = post_frames
                       
            if statistics and frames_required < 5:
                stats = 'Software version: {}\nSensitivity: {}\nTotal Frames: {}\nPre Movement Frames: {}\nPost Movement Frames: {}'.format(
                        __version__, sensitivity, frames_written, pre_frames, post_frames)
 
                buffered_frame = put_text(buffered_frame, stats, statistics_bgr)


            if frames_required > 0:
                if display:
                    cv2.imshow('Recorded Data', buffered_frame)
                if not mp4.is_open():
                    writer = mp4.open()
                    log.info('Opening {name}...'.format(name=mp4.get_filename()))
                jpg_frame = buffered_frame
                frames_required -= 1
                frames_written += 1
                writer.write(buffered_frame)
            else:
                if mp4.is_open():
                    journal.write('Closing {name}'.format(name=mp4.get_filename()))
                    # Write last frame. 
                    writer.write(buffered_frame)
                    mp4.close()
                    jpg_frame = buffered_frame
                    write_jpg(jpg_frame)
                    frames_written = 0
                    if display:
                        cv2.destroyWindow('Recorded Data')
                    if not command == "None":
                        cmd = command.replace('<MP4>', mp4.get_filename())
                        log.info('Command after replace is:{}'.format(cmd))
                        run_cmd(cmd)
                    else:
                        log.info('Command not run')

                    jpg_frame = None
                    movement_flag = False
                    version += 1
                    log.info('PID: {}'.format(os.getpid()))
            """
              Return image buffer
                :param data: Send image data back
            """
            cam.returnFrameBuffer(data)
            # print('returnBuffer')

        # Closing down.
        # cap.release()
        cam.stopCamera()
        log.info('Closing camera...')
        # cap.stopCamera()
        # cap.closeCamera()
        cv2.destroyAllWindows()

        # Update ini file.
        parser = configparser.ConfigParser()
        parser.read('version.ini')
        parser.set('MP4', 'version', str(mp4.get_version()))
        fp = open('version.ini', 'w')
        parser.write(fp)
        fp.close()

        if mp4.is_open():
            log.info('Close open MP4 file.')
            mp4.close()

    cam.closeCamera()
    log.info('Exit Motion.')
    sys.exit(0)
