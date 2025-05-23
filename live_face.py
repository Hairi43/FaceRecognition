import cv2
import numpy
from pathlib import Path
import copy

import sys
sys.path.append('.')
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('api')

import yaml
import cv2
import numpy as np
import torch
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
from core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler

with open('config/model_conf.yaml') as f:
    model_conf = yaml.load(f, Loader=yaml.FullLoader)


# Open the default camera
# cam = cv2.VideoCapture(0)

# Get the default frame width and height
# frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))




# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))


class LiveFace:

    def __init__(self, video_source=None, draw_crop=False, draw_face=False, draw_landmarks=False, gamma_corr=0.0, level_of_acceptance=0.70, clahe=False, on_no_face_detected=None):
        self.source = video_source
        self.draw_crop = draw_crop
        self.draw_face = draw_face
        self.draw_landmarks = draw_landmarks
        self.gamma_corr = gamma_corr
        self.level_of_acceptance = level_of_acceptance
        self.clahe = clahe
        self.on_no_face_detected = on_no_face_detected

    def draw_rectangle_on_face(self, dets, frame):
            box = dets[0]
            box = list(map(int, box))
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)    

    def gamma_correction(self, image, gamma):
        ## [changing-contrast-brightness-gamma-correction]
        lookUpTable = np.empty((1,256), np.uint8)
        for i in range(256):
            lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

        res = cv2.LUT(image, lookUpTable)
        ## [changing-contrast-brightness-gamma-correction]

        # img_gamma_corrected = cv2.hconcat([image, res])
        return res
        # cv2.imshow("Gamma correction", img_gamma_corrected)
        # cv2.waitKey()
    
    # def draw_lines(self, frame):
    #     # kadrowanie
    #     start_point_v1 = (450, 0)
    #     end_point_v1 = (450, 720)
    #     start_point_v2 = (800, 0)
    #     end_point_v2 = (800, 720)
    #     start_point_h1 = (0, 160)
    #     end_point_h1 = (1280, 160)
    #     start_point_h2 = (0, 600)
    #     end_point_h2 = (1280, 600)

    #     # Green color in BGR
    #     color = (0, 0, 255)

    #     # Line thickness of 9 px
    #     thickness = 4

    #     # Using cv2.line() method
    #     # Draw a diagonal green line with thickness of 9 px
    #     frame = cv2.line(frame, start_point_v1, end_point_v1, color, thickness)
    #     frame = cv2.line(frame, start_point_v2, end_point_v2, color, thickness)
    #     frame = cv2.line(frame, start_point_h1, end_point_h1, color, thickness)
    #     frame = cv2.line(frame, start_point_h2, end_point_h2, color, thickness)
    #     return frame


    def show_score(self, score, possible_face_image, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (00, 200)
        org_2 = (00, 30)
        fontScale = 1
        color_red = (0, 0, 255)
        color_green = (0, 255, 0)
        thickness = 2

        filename = Path(possible_face_image).name

        if score > self.level_of_acceptance:
            frame = cv2.putText(frame, f"{score:.5f}", org, font, fontScale, 
                                color_green, thickness, cv2.LINE_AA, False)
            frame = cv2.putText(frame, f"{filename}", org_2, font, fontScale, 
                                (0,255,255), thickness, cv2.LINE_AA, False)
        else:
            frame = cv2.putText(frame, f"{score:.5f}", org, font, fontScale, 
                                color_red, thickness, cv2.LINE_AA, False)
            frame = cv2.putText(frame, f"{filename}", org_2, font, fontScale, 
                                (0,255,255), thickness, cv2.LINE_AA, False)
        return frame
    

    def run(self):
        """
            Facex detection
        """

        # common setting for all models, need not modify.
        model_path = 'models'

        # setting device on GPU if available, else CPU
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        device ='cpu'

        # face detection model setting.
        scene = 'non-mask'
        model_category = 'face_detection'
        model_name =  model_conf[scene][model_category]
        logger.info('Start to load the face detection model...')
        try:
            faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
            model, cfg = faceDetModelLoader.load_model()

            faceDetModelHandler = FaceDetModelHandler(model, device, cfg)
        except Exception as e:
            logger.error('Falied to load face detection Model.')
            logger.error(e)
            sys.exit(-1)
        else:
            logger.info('Success!')

        # face landmark model setting.
        model_category = 'face_alignment'
        model_name =  model_conf[scene][model_category]
        logger.info('Start to load the face landmark model...')
        try:
            faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)

            model, cfg = faceAlignModelLoader.load_model()

            faceAlignModelHandler = FaceAlignModelHandler(model, device, cfg)
        except Exception as e:
            logger.error('Failed to load face landmark model.')
            logger.error(e)
            sys.exit(-1)
        else:
            logger.info('Success!')

        # face recognition model setting.
        model_category = 'face_recognition'
        model_name =  model_conf[scene][model_category]    
        logger.info('Start to load the face recognition model...')
        try:
            faceRecModelLoader = FaceRecModelLoader(model_path, model_category, model_name)


            model, cfg = faceRecModelLoader.load_model()
            model = model.module.cpu() # added

            faceRecModelHandler = FaceRecModelHandler(model, device, cfg)
        except Exception as e:
            logger.error('Failed to load face recognition model.')
            logger.error(e)
            sys.exit(-1)
        else:
            logger.info('Success!')

        face_cropper = FaceRecImageCropper()




        """
            Pętla wideo
        """


        # video = cv2.VideoCapture('output2.mp4')
        video = cv2.VideoCapture(self.source)
        cv2.namedWindow("video", cv2.WINDOW_NORMAL)

        if video is None:
            print('Warning: unable to open video source: ', video)
            return

        dets = numpy.array([])

        # measurments
        score = 0
        # avg_score = 0
        # measurment_counter = 0

        # if found face with most probability of being the same people
        checked_all_faces = False
        max_score = -1
        possible_face_image = ''

        folder_dir = 'imagesDB'
        images = Path(folder_dir).glob('*.jpg')

        with open("scoresDB/new_score.txt", 'w') as f:
            while video.isOpened():
                ret, frame = video.read()

                if not ret:
                    break

                #print(f"frame shape original {frame.shape}")

                print(f"self.clahe = {self.clahe}")

                if self.clahe:
                    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                    lab_planes = list(cv2.split(lab))
                    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
                    lab_planes[0] = clahe.apply(lab_planes[0])
                    lab = cv2.merge(lab_planes)
                    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

                    #print(f"frame shape after {frame.shape}")

                # if self.draw_crop:
                #     frame = self.draw_lines(frame)

                if self.gamma_corr > 0.001:
                    frame = self.gamma_correction(frame, self.gamma_corr)

                frame_draw = copy.deepcopy(frame)


                if checked_all_faces == False:
                    for image in images:
                        print("currently processing image", image)
                        try:
                            dets = faceDetModelHandler.inference_on_image(frame)
                            dets = numpy.append(dets, faceDetModelHandler.inference_on_image(cv2.imread(image)))
                            print(dets.shape)

                            if dets.shape[0] == 10:
                                dets = dets.reshape(2, 5)
                                if self.draw_face:
                                    self.draw_rectangle_on_face(dets, frame_draw)
                        except Exception as e:
                                logger.error('Face detection failed!')
                                logger.error(e)


                        # frame rate - 10 fps - prawie...
                        # if counter % 20 != 0:
                        """
                            pipeline
                        """

                        try:
                            if dets.shape[0] == 2:
                                face_nums = dets.shape[0]
                                # face_nums = []
                                if face_nums != 2:
                                    logger.info('Input image should contain two faces to compute similarity!')
                                feature_list = []
                                for i in range(face_nums):
                                    landmarks = faceAlignModelHandler.inference_on_image(frame, dets[i])
                                    landmarks_list = []
                                    for (x, y) in landmarks.astype(np.int32):
                                        landmarks_list.extend((x, y))
                                        if i == 0 and self.draw_landmarks:
                                            cv2.circle(frame_draw, (x, y), 2, (0, 255, 0),-1)
                                    cropped_image = face_cropper.crop_image_by_mat(frame, landmarks_list)
                                    feature = faceRecModelHandler.inference_on_image(cropped_image)
                                    feature_list.append(feature)
                                score = np.dot(feature_list[0], feature_list[1])
                                logger.info('The similarity score of two faces: %f' % score)

                                # if found more probable face to video then save it
                                if score > max_score:
                                    max_score = score
                                    print("image_name ", image)
                                    possible_face_image = str(image)
                                    print(possible_face_image)
                        except Exception as e:
                            logger.error('Pipeline failed!')
                            logger.error(e)
                            sys.exit(-1)
                        else:
                            logger.info('Success!')
                    checked_all_faces = True
                elif possible_face_image != '':
                    print("currently processing best image", possible_face_image)
                    # print("gamma level", self.gamma_correction)
                    try:
                        dets = faceDetModelHandler.inference_on_image(frame)
                        dets = numpy.append(dets, faceDetModelHandler.inference_on_image(cv2.imread(possible_face_image)))

                        if dets.shape[0] == 10:
                            dets = dets.reshape(2, 5)
                            if self.draw_face:
                                self.draw_rectangle_on_face(dets, frame_draw) # drawing lines around face
                    except Exception as e:
                            logger.error('Face detection failed!')
                            logger.error(e)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            org = (00, 60)
                            fontScale = 1
                            color = (255, 255, 0)
                            thickness = 2
                            frame = cv2.putText(frame, f"nie znaleziono twarzy", org, font, fontScale, 
                                                color, thickness, cv2.LINE_AA, False)


                    # frame rate - 10 fps - prawie...
                    # if counter % 20 != 0:
                    """
                        pipeline
                    """

                    try:
                        if dets.shape[0] == 2:
                            face_nums = dets.shape[0]
                            # face_nums = []
                            if face_nums != 2:
                                logger.info('Input image should contain two faces to compute similarity!')
                            feature_list = []
                            for i in range(face_nums):
                                landmarks = faceAlignModelHandler.inference_on_image(frame, dets[i])
                                landmarks_list = []
                                for (x, y) in landmarks.astype(np.int32):
                                    landmarks_list.extend((x, y))
                                    if i == 0 and self.draw_landmarks:
                                        cv2.circle(frame_draw, (x, y), 2, (0, 255, 0),-1) # draw dots on face
                                cropped_image = face_cropper.crop_image_by_mat(frame, landmarks_list)
                                feature = faceRecModelHandler.inference_on_image(cropped_image)
                                feature_list.append(feature)

                            score = np.dot(feature_list[0], feature_list[1])
                            # avg_score += score

                            f.write(f"{score:.5f}\n")

                            logger.info(f'The similarity score of two faces: {score:.5f}')
                    except Exception as e:
                        logger.error('Pipeline failed!')
                        logger.error(e)
                        sys.exit(-1)
                    else:
                        logger.info('Success!')
                else:
                    if self.on_no_face_detected:
                        self.on_no_face_detected()
                    break

                    # measurment_counter += 1


                # Write the frame to the output file
                # out.write(frame)

                # Display the captured frame
                # cv2.imshow('Camera', frame)

                # frame
                # if score > self.level_of_acceptance:
                #     frame = cv2.putText(frame, str(score), org, font, fontScale, 
                #                         color_green, thickness, cv2.LINE_AA, False)
                # else:
                #     frame = cv2.putText(frame, str(score), org, font, fontScale, 
                #                         color_red, thickness, cv2.LINE_AA, False)

                frame_draw = self.show_score(score, possible_face_image, frame_draw)
                    
                cv2.imshow('video', frame_draw)
                
                # Press 'q' to exit the loop
                if cv2.waitKey(1) == ord('q'):
                    break
        
        # if measurment_counter != 0:
        #     avg_score /= measurment_counter
        # else:
        #     print(f"measurment counter equals zero")

        
        # with open("scoresDB/new_score.txt", 'a') as f:
        #     f.write(str(avg_score))


        # Release the capture and writer objects
        # cam.release()
        video.release()
        # out.release()
        cv2.destroyAllWindows()




    def run_live(self):
        """
            Facex detection
        """

        # common setting for all models, need not modify.
        model_path = 'models'

        # setting device on GPU if available, else CPU
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')

        device ='cpu'

        # face detection model setting.
        scene = 'non-mask'
        model_category = 'face_detection'
        model_name =  model_conf[scene][model_category]
        logger.info('Start to load the face detection model...')
        try:
            faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
            model, cfg = faceDetModelLoader.load_model()
            faceDetModelHandler = FaceDetModelHandler(model, device, cfg)
        except Exception as e:
            logger.error('Falied to load face detection Model.')
            logger.error(e)
            sys.exit(-1)
        else:
            logger.info('Success!')

        # face landmark model setting.
        model_category = 'face_alignment'
        model_name =  model_conf[scene][model_category]
        logger.info('Start to load the face landmark model...')
        try:
            faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
            model, cfg = faceAlignModelLoader.load_model()
            faceAlignModelHandler = FaceAlignModelHandler(model, device, cfg)
        except Exception as e:
            logger.error('Failed to load face landmark model.')
            logger.error(e)
            sys.exit(-1)
        else:
            logger.info('Success!')

        # face recognition model setting.
        model_category = 'face_recognition'
        model_name =  model_conf[scene][model_category]    
        logger.info('Start to load the face recognition model...')
        try:
            faceRecModelLoader = FaceRecModelLoader(model_path, model_category, model_name)
            model, cfg = faceRecModelLoader.load_model()

            model = model.module.cpu() # added

            faceRecModelHandler = FaceRecModelHandler(model, device, cfg)
        except Exception as e:
            logger.error('Failed to load face recognition model.')
            logger.error(e)
            sys.exit(-1)
        else:
            logger.info('Success!')

        face_cropper = FaceRecImageCropper()




        """
            Pętla wideo
        """


        # video = cv2.VideoCapture('output2.mp4')
        cam = cv2.VideoCapture(self.source)
        cv2.namedWindow("live", cv2.WINDOW_NORMAL)

        if cam is None:
            print('Warning: unable to open video source: ', cam)
            return

        counter = 0

        dets = numpy.array([])

        score = 0

        # if found face with most probability of being the same people
        checked_all_faces = False
        max_score = -1
        possible_face_image = ''

        folder_dir = 'imagesDB'
        images = Path(folder_dir).glob('*.jpg')

        # measurments
        # avg_score = 0
        # measurment_counter = 0

        with open("scoresDB/new_score.txt", 'w') as f:
            while True:
                ret, frame = cam.read()

                if not ret:
                    break

                if self.clahe:
                    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                    lab_planes = list(cv2.split(lab))
                    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
                    lab_planes[0] = clahe.apply(lab_planes[0])
                    lab = cv2.merge(lab_planes)
                    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

                # if self.draw_crop:
                #     frame = self.draw_lines(frame)

                if self.gamma_corr > 0.001:
                    frame = self.gamma_correction(frame, self.gamma_corr)

                frame_draw = copy.deepcopy(frame)


                if checked_all_faces == False:
                    for image in images:
                        print("currently processing image", image)
                        try:
                            dets = faceDetModelHandler.inference_on_image(frame)
                            dets = numpy.append(dets, faceDetModelHandler.inference_on_image(cv2.imread(image)))

                            cv2.imshow('video', frame)

                            if dets.shape[0] == 10:
                                dets = dets.reshape(2, 5)
                                if self.draw_face:
                                    self.draw_rectangle_on_face(dets, frame_draw) # drawing lines around face
                        except Exception as e:
                                logger.error('Face detection failed!')
                                logger.error(e)

                                font = cv2.FONT_HERSHEY_SIMPLEX
                                org = (00, 60)
                                fontScale = 1
                                color = (255, 255, 0)
                                thickness = 2
                                frame = cv2.putText(frame, f"nie znaleziono twarzy", org, font, fontScale, 
                                                    color, thickness, cv2.LINE_AA, False)


                        # frame rate - 10 fps - prawie...
                        # if counter % 20 != 0:
                        """
                            pipeline
                        """

                        try:
                            if dets.shape[0] == 2:
                                face_nums = dets.shape[0]
                                # face_nums = []
                                if face_nums != 2:
                                    logger.info('Input image should contain two faces to compute similarity!')
                                feature_list = []
                                for i in range(face_nums):
                                    landmarks = faceAlignModelHandler.inference_on_image(frame, dets[i])
                                    landmarks_list = []
                                    for (x, y) in landmarks.astype(np.int32):
                                        landmarks_list.extend((x, y))
                                        if i == 0 and self.draw_landmarks:
                                            cv2.circle(frame_draw, (x, y), 2, (0, 255, 0),-1) # draw dots on face
                                    cropped_image = face_cropper.crop_image_by_mat(frame, landmarks_list)
                                    feature = faceRecModelHandler.inference_on_image(cropped_image)
                                    feature_list.append(feature)
                                score = np.dot(feature_list[0], feature_list[1])
                                logger.info('The similarity score of two faces: %f' % score)

                                # if found more probable face to video then save it
                                if score > max_score:
                                    max_score = score
                                    print("image_name ", image)
                                    possible_face_image = str(image)
                                    print(possible_face_image)
                        except Exception as e:
                            logger.error('Pipeline failed!')
                            logger.error(e)
                            sys.exit(-1)
                        else:
                            logger.info('Success!')
                    checked_all_faces = True
                elif possible_face_image != '':
                    print("currently processing best image", possible_face_image)
                    try:
                        dets = faceDetModelHandler.inference_on_image(frame)
                        dets = numpy.append(dets, faceDetModelHandler.inference_on_image(cv2.imread(possible_face_image)))

                        cv2.imshow('video', frame)

                        if dets.shape[0] == 10:
                            dets = dets.reshape(2, 5)
                            if self.draw_face:
                                self.draw_rectangle_on_face(dets, frame_draw) # drawing lines around face
                    except Exception as e:
                            logger.error('Face detection failed!')
                            logger.error(e)

                            font = cv2.FONT_HERSHEY_SIMPLEX
                            org = (00, 60)
                            fontScale = 1
                            color = (255, 255, 0)
                            thickness = 2
                            frame = cv2.putText(frame, f"nie znaleziono twarzy", org, font, fontScale, 
                                                color, thickness, cv2.LINE_AA, False)


                    # frame rate - 10 fps - prawie...
                    # if counter % 20 != 0:
                    """
                        pipeline
                    """

                    try:
                        if dets.shape[0] == 2:
                            face_nums = dets.shape[0]
                            # face_nums = []
                            if face_nums != 2:
                                logger.info('Input image should contain two faces to compute similarity!')
                            feature_list = []
                            for i in range(face_nums):
                                landmarks = faceAlignModelHandler.inference_on_image(frame, dets[i])
                                landmarks_list = []
                                for (x, y) in landmarks.astype(np.int32):
                                    landmarks_list.extend((x, y))
                                    if i == 0 and self.draw_landmarks:
                                        cv2.circle(frame_draw, (x, y), 2, (0, 255, 0),-1) # draw dots on face
                                cropped_image = face_cropper.crop_image_by_mat(frame, landmarks_list)
                                feature = faceRecModelHandler.inference_on_image(cropped_image)
                                feature_list.append(feature)

                            score = np.dot(feature_list[0], feature_list[1])
                            # measurment_counter += 1
                            # avg_score += score

                            f.write(f"{score:.5f}\n")

                            logger.info(f'The similarity score of two faces: {score:.5f}')
                    except Exception as e:
                        logger.error('Pipeline failed!')
                        logger.error(e)
                        sys.exit(-1)
                    else:
                        logger.info('Success!')
                else:
                    if self.on_no_face_detected:
                        self.on_no_face_detected()
                    break
                        # measurment_counter += 1
                    # Write the frame to the output file
                    # out.write(frame)

                    # Display the captured frame
                    # cv2.imshow('Camera', frame)


                frame_draw = self.show_score(score, possible_face_image,frame_draw)

                cv2.imshow('video', frame_draw)


                # bboxs = dets
                # for box in bboxs:
                #     box = list(map(int, box))
                #     cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)



                # Write the frame to the output file
                # out.write(frame)

                # Display the captured frame
                # cv2.imshow('Camera', frame)

                # Press 'q' to exit the loop
                if cv2.waitKey(1) == ord('q'):
                    break
        
        # if measurment_counter != 0:
        #     avg_score /= measurment_counter
        # else:
        #     print(f"measurment counter equals zero")

        # Release the capture and writer objects
        # cam.release()
        cam.release()
        # out.release()
        cv2.destroyAllWindows()