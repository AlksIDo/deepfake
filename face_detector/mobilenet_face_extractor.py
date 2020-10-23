import tensorflow as tf
import numpy as np
import cv2


class FaceExtractor:

    def __init__(self, model_path):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        cm = detection_graph.as_default()
        cm.__enter__()

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(graph=detection_graph, config=config)
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.boxes_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.scores_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    def get_mobilenet_faces(self, my_frames, my_idxs, confidence=0.9):
        im_heights, im_widths, imgs = [], [], []
        for frame in my_frames:
            (im_height, im_width) = frame.shape[:-1]
            imgs.append(frame)
            im_heights.append(im_height)
            im_widths.append(im_widths)

        imgs = np.array(imgs)
        (boxes, scores_) = self.sess.run(
            [self.boxes_tensor, self.scores_tensor],
            feed_dict={self.image_tensor: imgs})

        processed_frames = []
        for i, frame_idx in zip(range(boxes.shape[0]), my_idxs):
            scores = scores_[i]
            indexes = np.where(scores > confidence)[0]
            if indexes.shape[0] == 0:
                processed_frames.append({
                    'frame_idx': frame_idx,
                    'faces': [],
                    'scores': [],
                })
                continue

            image = imgs[i]
            processed_faces, processed_scores = [], []
            for index in indexes:
                box = boxes[i][index]
                ymin, xmin, ymax, xmax = box
                (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                              ymin * im_height, ymax * im_height)
                left, right, top, bottom = int(left), int(right), int(top), int(bottom)

                processed_faces.append(image[max([0, top - 20]):bottom + 20, max([0, left - 20]):right + 20])
                processed_scores.append(scores[index])

            processed_frames.append({
                'frame_idx': frame_idx,
                'faces': processed_faces,
                'scores': processed_scores,
            })

        return processed_frames
