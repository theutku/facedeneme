import cv2
import numpy as np
import os
import subprocess
from shutil import copy2
import math

from opencv.cascade.downloadbase import CascadeImageProcessor
from opencv.cascade.paths import CascadeDirs


class HaarCascadeBase(CascadeImageProcessor):

    def __init__(self, download_dir='downloads', cascade_dir='cascadedata'):
        super().__init__(download_dir=download_dir)
        self.positive_file_count = 0
        self.info_file = ''

        self.cascade_dirs = CascadeDirs(cascade_dir=cascade_dir)

    def printVideoMessage(self, message='', key_message=''):
        if message == '':
            print('Starting Video Feed...')
            print('Press ESC to quit')
        else:
            print(message)
            print(key_message)

    def loadCascadeFile(self, cascade_file):

        if type(cascade_file) is list:
            cascades = []
            for cascade in cascade_file:
                cas = cv2.CascadeClassifier(cascade)
                cascades.append(cas)
            return cascades
        elif type(cascade_file) is str:
            cas = cv2.CascadeClassifier(cascade_file)
            return cas

    def display_faces(self, cascade_files, videoSource=0):
        cap = cv2.VideoCapture(videoSource)
        self.printVideoMessage()

        face_cascade = self.loadCascadeFile(cascade_files)
        # cascades = self.loadCascadeFile(cascade_files)
        # face_cascade = cascades[0]
        # eye_cascade = cascades[1]

        while True:
            _, frame = cap.read()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Original Video Feed', frame)

            faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

            cv2.imshow('Faces', frame)

            if cv2.waitKey(27) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def create_desc_files(self):
        if os.path.exists('bg.txt'):
            os.remove('bg.txt')
        if os.path.exists('info.dat'):
            os.remove('info.dat')

        for neg in os.listdir(self.download_dirs.neg):
            line = os.path.join(self.download_dirs.neg, neg) + '\n'
            with open('bg.txt', 'a') as f:
                f.write(line)

        for pos in os.listdir(self.cascade_dirs.info):
            line = pos + ' 1 0 0 50 50\n'
            with open('cascadedata/info/info.lst', 'a') as f:
                f.write(line)

    def form_positive_vector(self, file_name, samples, width, height):
        vector_file = os.path.join(self.cascade_dirs.main, file_name + '.vec')
        self.vector_file = vector_file
        print('Creating positive vector file...')

        subprocess.call(
            'opencv_createsamples -info {0} -num {1} -w {2} -h {3} -vec {4}'.format('cascadedata/info/info.lst', samples, width, height, vector_file), shell=True)

    def train_classifier(self, output_dir='cascade/data', vec_name='positives', num_stages=10, vec_width=20, vec_height=20, width=20, height=20):
        total_pos = len(os.listdir(self.cascade_dirs.info)) - 1
        vec_samples = total_pos
        self.form_positive_vector(
            vec_name, vec_samples, width=vec_width, height=vec_height)

        num_pos = math.floor(total_pos * 0.7)
        num_neg = math.floor(num_pos * 0.7)

        print('Training Cascade Classifier...')
        subprocess.call(
            'opencv_traincascade -data {0} -vec {1} -bg bg.txt -numPos {2} -numNeg {3} -minHitRate 0.998 -maxFalseAlarmRate 0.5 -numStages {4} -w {5} -h {6}'.format(output_dir, self.vector_file, num_pos, num_neg, num_stages, width, height), shell=True)
