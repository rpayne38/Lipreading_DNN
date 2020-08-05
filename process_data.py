#This program preprocesses video and text files from the LRS3 dataset and generates clipped video files, 
#cropped to the speakers mouth region, with the label included in the filename of the generated mp4.
#Requires the dlib facial landmark predictor

import cv2
import dlib
import numpy as np
from imutils import face_utils
import os
import math
from random import shuffle

input_dir="videos"
output="output"
facial_landmark_predictor="shape_predictor_68_face_landmarks.dat"
fps=25
num_samples=600

WORDS=["SOMETHING", "THEN", "WORLD", "BECAUSE", "YOU", "PEOPLE", "REALLY", "ABOUT", "LIKE", "WE"]
COUNT=[0]*len(WORDS)

#fetch and shuffle list of txt files from input dir
def sortFiles(input):
    print("Getting List of Files...")
    listOfFiles=list()
    listOfTxt=list()
    for (dirpath, dirnames, filenames) in os.walk(input):
            listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    for file in listOfFiles:
        if file.endswith(".txt"):
            listOfTxt.append(file)
        else:
            pass

    shuffle(listOfTxt)

    return listOfTxt

#look for each word in the input file, get the start and end frames, pass them to writevid()
def time2frame(file,WORDS,detector,predictor):
    for i,word in enumerate(WORDS):
        if COUNT[i]<=num_samples:
            with open(file) as f:
                for num, line in enumerate(f,1):
                    if num<5:
                        pass
                    elif f' {word} ' in f' {line} ':
                        WORD,START,END,ASD=line.split(" ")
                        START=math.floor(float(START)*fps)
                        END=math.ceil(float(END)*fps)
                        location,extension=file.split(".")
                        vid_file=f"{location}.mp4"
                        COUNT[i]+=1
                        writevid(vid_file,WORD,START,END,i,detector,predictor)
            f.close() 
        else:
            pass

    return

#crop video to speaker's mouth and word boundaries
def writevid(file,word,start,end,i,detector,predictor):
    #Initialise video capture
    frame_num=1
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap=cv2.VideoCapture(file)

    #create unique video file name
    num_files=0
    filename=f"{output}/{word}/{word}_{num_files}.mp4"
    while os.path.isfile(filename)==True:
        num_files+=1
        filename=f"{output}/{word}/{word}_{num_files}.mp4"

    #initialise video writer to save as [224,224,3]
    writer = cv2.VideoWriter(filename, fourcc, fps,(224, 224), True)

    #loop through frames until the start of the word is reached
    while(cap.isOpened):
        ret,frame=cap.read()
        if ret==True:
            if frame_num in range(start,end+1):  

                # detect faces in a grayscale image
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 1)

                #if faces are detected
                if rects:

                    # loop over the face detections
                    for (num, rect) in enumerate(rects):
                        # determine the facial landmarks for the face region, then
                        # convert the landmark (x, y)-coordinates to a NumPy array
                        shape = predictor(gray, rect)
                        shape = face_utils.shape_to_np(shape)

                        # extract the ROI of the mouth (point 48-68)
                        (x, y, w, h) = cv2.boundingRect(np.array([shape[48:68]]))
                        #crop frame and resize to [224,224]
                        roi = frame[y:y+h, x:x+w]
                        image = cv2.resize(roi, (224,224))

                    #write the image
                    writer.write(image)

                #if no face is detected for one of the frames, delete the file and decrement the counter
                else:
                    os.remove(filename)
                    COUNT[i]-=1
                    break
        
            #break out of loop once end of word has breen reached
            elif frame_num>end+1:
                break

            else:
                pass

        frame_num+=1

    cap.release()
    writer.release()
    return

#Print Progress bar on screen
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()     

    return

def main():
    #make directory for each word
    os.mkdir(output)
    for i in WORDS:
        os.mkdir(f"{output}/{i}")

    #initialise face detector here so that it isn't called multiple times
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(facial_landmark_predictor)

    #get list of txt files to be processed
    listOfTxt=sortFiles(input_dir)

    # Initial call to print 0% progress
    l=len(listOfTxt)
    printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', decimals = 4, length = 200)

    #loop through txt list, looking for listed words and generate the labelled video files
    for i,file in enumerate(listOfTxt):
        time2frame(file,WORDS,detector,predictor)
        printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', decimals = 4, length = 200)

if __name__=="__main__":
    main()