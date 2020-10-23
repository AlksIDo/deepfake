import os
from os.path import exists
from os import makedirs

import cv2
import torch
import numpy as np

from .utils import VideoReader, isotropically_resize_image, make_square_image
from .blaze_face import BlazeFace
from .blaze_face_extractor import FaceExtractor as BlazeFaceExtractor
from .mobilenet_face_extractor import FaceExtractor as MobileNetFaceExtractor


print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ROOT = os.path.dirname(os.path.dirname(__file__))

DEFAULT_WEIGHTS_PATH = os.path.join(ROOT, 'models', 'blazeface.pth')
DEFAULT_ANCHORS_PATH = os.path.join(ROOT, 'models', 'anchors.npy')
DEFAULT_MOBILENET_MODEL_PATH = os.path.join(ROOT, 'models', 'frozen_inference_graph_face.pb')


class FaceDetector:

    def __init__(
            self,
            weights_path=DEFAULT_WEIGHTS_PATH,
            anchors_path=DEFAULT_ANCHORS_PATH,
            mobilenet_model_path=DEFAULT_MOBILENET_MODEL_PATH,
            frames_per_video=30,
            output_size=224,
    ):
        facedet = BlazeFace().to(device)
        facedet.load_weights(weights_path)
        facedet.load_anchors(anchors_path)
        _ = facedet.train(False)

        self.blaze_face_extractor = BlazeFaceExtractor(facedet)
        self.mobilenet_face_extractor = MobileNetFaceExtractor(mobilenet_model_path)

        self.frames_per_video = frames_per_video
        self.video_reader = VideoReader()
        self.video_read_fn = lambda x: self.video_reader.read_frames(x, num_frames=self.frames_per_video)

        self.output_size = output_size

    def get_faces(self, my_frames, my_idxs,
                  blaze_confidence1=0.9, blaze_confidence2=0.9,
                  mbnet_confidence1=0.9, mbnet_confidence2=0.6):

        mbnet_confidences = [mbnet_confidence1, mbnet_confidence2]
        blaze_confidences = [blaze_confidence1, blaze_confidence2]

        try:
            blaze_frames = self.get_blaze_faces(my_frames, my_idxs)
        except Exception as e:
            print(f'[Blaze Face Warning]: {e}')
            blaze_frames = []

        try:
            mbnet_frames = self.get_mobilenet_faces(my_frames, my_idxs, confidence=0.6)
        except Exception as e:
            print(f'[MobileNet Face Warning]: {e}')
            mbnet_frames = []

        faces = []
        if len(blaze_frames) != 0 and len(mbnet_frames) != 0:
            for blaze_frame, mbnet_frame in zip(blaze_frames, mbnet_frames):
                blaze_existed = False
                blaze_faces = self.choose_faces(blaze_frame, blaze_confidences)
                if blaze_faces:
                    faces.extend(blaze_faces)
                    blaze_existed = True
                if not blaze_existed:
                    faces.extend(self.choose_faces(mbnet_frame, mbnet_confidences))
        elif len(blaze_frames) != 0:
            for frame in blaze_frames:
                faces.extend(self.choose_faces(frame, blaze_confidences))
        elif len(mbnet_frames) != 0:
            for frame in mbnet_frames:
                faces.extend(self.choose_faces(frame, mbnet_confidences))

        return faces

    def choose_faces(self, frame, confidences):
        faces = []
        for i, (face, score) in enumerate(zip(frame['faces'], frame['scores'])):
            if i == 2:
                break
            if score > confidences[i]:
                faces.append(
                    cv2.resize(face, (self.output_size, self.output_size), interpolation=cv2.INTER_CUBIC)
                )
        return faces

    def get_blaze_faces(self, my_frames, my_idxs):
        faces = []
        for processed_frame in self.blaze_face_extractor.process_frames(my_frames, my_idxs):
            faces.append({
                'frame_idx': processed_frame['frame_idx'],
                'faces': processed_frame['faces'],
                'scores': processed_frame['scores'],
            })
        return faces

    def get_mobilenet_faces(self, my_frames, my_idxs, confidence=0.9):
        return self.mobilenet_face_extractor.get_mobilenet_faces(my_frames, my_idxs, confidence=confidence)


#         self.face_extractor.keep_only_best_face(faces)
#
#         if len(faces) > 0:
#             # NOTE: When running on the CPU, the batch size must be fixed
#             # or else memory usage will blow up. (Bug in PyTorch?)
#             x = np.zeros((self.frames_per_video, self.output_size, self.output_size, 3), dtype=np.uint8)
#
#             # If we found any faces, prepare them for the model.
#             n = 0
#             for frame_data in faces:
#                 for face in frame_data["faces"]:
#                     # Resize to the model's required input size.
#                     # We keep the aspect ratio intact and add zero
#                     # padding if necessary.
# #                     resized_face = isotropically_resize_image(face, self.input_size)
#                     resized_face = cv2.resize(face, (self.output_size, self.output_size), interpolation=cv2.INTER_AREA)
# #                     resized_face = make_square_image(resized_face)
#                     if n < self.frames_per_video:
#                         x[n] = resized_face
#                         n += 1
#                     else:
#                         pass
#         #                         print("WARNING: have %d faces but batch size is %d" % (n, batch_size))
#
#         # Test time augmentation: horizontal flips.
#         # TODO: not sure yet if this helps or not
#         # x[n] = cv2.flip(resized_face, 1)
#         # n += 1
#         return x, n
