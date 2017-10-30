import os
import cv2
import dlib
import numpy as np
import argparse
#from wide_resnet import WideResNet
from TYY_model import TYY_2stream,TYY_1stream
import sys
import timeit
from moviepy.editor import VideoFileClip

def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from video, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    args = parser.parse_args()
    return args


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


def main():
    args = get_args()
    depth = args.depth
    k = args.width
    weight_file = args.weight_file

    if not weight_file:
        #weight_file = os.path.join("pretrained_models", "weights.18-4.06.hdf5")
        weight_file = os.path.join("models", "TYY_1stream.h5")

    # for face detection
    detector = dlib.get_frontal_face_detector()

    # load model and weights
    img_size = 64
    #model = WideResNet(img_size, depth=depth, k=k)()
    model = TYY_1stream(img_size)()
    model.load_weights(weight_file)

    clip = VideoFileClip('mewtwo.mp4') # can be gif or movie

    img_idx = 0
    detected = '' #make this not local variable
    time_detection = 0
    time_network = 0
    time_plot = 0
    #scale = 0.7 #scaling the input image
    for img in clip.iter_frames():
        img_idx = img_idx + 1
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #input_img = cv2.resize(input_img,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
        img_h, img_w, _ = np.shape(input_img)

        
        if img_idx==1 or img_idx%1 == 0:
            
            # detect faces using dlib detector
            detected = detector(input_img, 1)
            faces = np.empty((len(detected), img_size, img_size, 3))

            start_time = timeit.default_timer()
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - 0.4 * w), 0)
                yw1 = max(int(y1 - 0.4 * h), 0)
                xw2 = min(int(x2 + 0.4 * w), img_w - 1)
                yw2 = min(int(y2 + 0.4 * h), img_h - 1)
                cv2.rectangle(input_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                faces[i,:,:,:] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
            elapsed_time = timeit.default_timer()-start_time
            time_detection = time_detection + elapsed_time
            
            start_time = timeit.default_timer()
            if len(detected) > 0:
                # predict ages and genders of the detected faces
                results = model.predict(faces)
                predicted_genders = results[0]
                ages = np.arange(0, 21).reshape(21, 1)
                predicted_ages = results[1].dot(ages).flatten()

            # draw results
            for i, d in enumerate(detected):
                label = "{}~{}, {}".format(int((predicted_ages[i]-1)*5),int(predicted_ages[i]*5),
                                        "F" if predicted_genders[i][0] > 0.5 else "M")
                draw_label(input_img, (d.left(), d.top()), label)
            elapsed_time = timeit.default_timer()-start_time
            time_network = time_network + elapsed_time
            
            
            #input_img = cv2.resize(input_img,None,fx=1/scale,fy=1/scale,interpolation=cv2.INTER_CUBIC)
            
            start_time = timeit.default_timer()
            cv2.imshow("result", input_img)
            elapsed_time = timeit.default_timer()-start_time
            time_plot = time_plot + elapsed_time
            
            
        else:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - 0.4 * w), 0)
                yw1 = max(int(y1 - 0.4 * h), 0)
                xw2 = min(int(x2 + 0.4 * w), img_w - 1)
                yw2 = min(int(y2 + 0.4 * h), img_h - 1)
                cv2.rectangle(input_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                faces[i,:,:,:] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

            # draw results
            for i, d in enumerate(detected):
                label = "{}~{}, {}".format(int((predicted_ages[i]-1)*4.7),int(predicted_ages[i]*4.7),
                                        "F" if predicted_genders[i][0] > 0.5 else "M")
                draw_label(input_img, (d.left(), d.top()), label)
            #input_img = cv2.resize(input_img,None,fx=1/scale,fy=1/scale,interpolation=cv2.INTER_CUBIC)
            cv2.imshow("result", input_img)
        
        #Show the time cost (fps)
        print('avefps_time_detection:',img_idx/time_detection)
        print('avefps_time_network:',img_idx/time_network)
        print('avefps_time_plot:',img_idx/time_plot)
        print('===============================')
        key = cv2.waitKey(30)

        if key == 27:
            break


if __name__ == '__main__':
    main()
