import grpc
from concurrent import futures
import time

import detector_pb2
import detector_pb2_grpc

import numpy as np
from collections import deque
from inference import load_config, load_model, inference_

class Detector(detector_pb2_grpc.DetectorServicer):

    def __init__(self):
        # self.model = None
        self.pose_results_dict = {}
        self.objects = []
        self.max_len = 60
        self.V = 17
        self.config = load_config()
        self.model = load_model(config=self.config)
        self.keypoint_score = np.ones(shape = (1, self.max_len, self.V), dtype=np.float16)
        self.keypoint = np.ones(shape = (1, self.max_len, self.V, 2), dtype=np.float16)
        self.label_map = [x.strip() for x in open("/home/bishal/Documents/thesis/pyskl/gRPC/aitdata_labels.txt").readlines()]



    def DetectTest(self, request, context):
        print("Received detection request for test")
    
        # Perform detection on the image and return results
        response = detector_pb2.ActionResult()
        for detection_result in request.detectionResult:
            for detection in detection_result.detection:
                action_class = detector_pb2.ActionClass()
                action_class.objectId = detection.objectId
                action_class.streamId = detection.streamId
                action_class.classId = 1 # Replace with your desired fake class label
                response.classes.append(action_class)

        # populate response with detected classes
        return response

    def Detect(self, request, context):
        print("Received detection request")
        batch = []
        objectIdsInBatch = []
        streamIdInBatch = []
        max_len = self.max_len

        h = 320
        w = 640
        fake_anno = dict(
            frame_dir='',
            label=-1,
            img_shape=(h, w),
            original_shape=(h, w),
            start_index=0,
            modality='Pose',
            total_frames=max_len)

        # Perform detection on the image and return results
        response = detector_pb2.ActionResult()
        for detection_result in request.detectionResult:
            for detection in detection_result.detection:
                objectId = detection.objectId
                streamId = detection.streamId
                if objectId not in self.pose_results_dict:
                    self.pose_results_dict[objectId] = deque(maxlen=max_len) # shape T x V x C
                pose_sequence = [] # shape V x C
                for i, point in enumerate(detection.points):
                    if i!=0: # skipping the neck (18 keypoints from trtpose, my model needs 17 keypoints)
                        pose_sequence.append([point.x, point.y])
                
                self.pose_results_dict[objectId].append(pose_sequence)

                if len(self.pose_results_dict[objectId]) == max_len:
                    print("Ready: ", objectId)
                    batch.append(self.pose_results_dict[objectId])
                    objectIdsInBatch.append(objectId)
                    streamIdInBatch.append(streamId)
                    # keypoint = np.array(self.pose_results_dict[objectId], dtype=np.float16)
                    # keypoint = np.expand_dims(keypoint, axis=0)
                    # # print(keypoint.shape)
                    # # print(keypoint)
                    # # print(self.keypoint_score.shape)
                    # fake_anno['keypoint'] = keypoint
                    # fake_anno['keypoint_score'] = self.keypoint_score
                    # # print(fake_anno)
                    # # scores = inference_(model = self.model,  data = fake_anno)
                    # # action_label = self.label_map[scores[0][0]]
                    # # classId = scores[0][0]
                    # # print(action_label)

                    # old_pose_estimation = self.pose_results_dict[objectId].popleft()

                    # fake_anno = dict(
                    #     frame_dir='',
                    #     label=-1,
                    #     img_shape=(h, w),
                    #     original_shape=(h, w),
                    #     start_index=0,
                    #     modality='Pose',
                    #     total_frames=max_len)

        if len(objectIdsInBatch) >= 1:
            keypoint = np.array(batch, dtype=np.float16)
            print(keypoint.shape)
            keypoint_score = np.ones(shape = (len(objectIdsInBatch), self.max_len, self.V), dtype=np.float16)
            print(keypoint_score.shape)
            # print(keypoint)
            # print(self.keypoint_score.shape)
            fake_anno['keypoint'] = keypoint
            fake_anno['keypoint_score'] = self.keypoint_score
            scores = inference_(model = self.model,  data = fake_anno)
            # print(scores.shape)
            # action_label = self.label_map[scores[0][0]]
            # classId = scores[0][0]
            # print(action_label)

        action_class = detector_pb2.ActionClass()
        action_class.objectId = 1
        action_class.streamId = 1
        # # print("stream Id: ", detection.streamId)
        # # print("Class Id: ", detection.objectId)
        action_class.classId = 1 # Replace with your desired fake class label
        response.classes.append(action_class)


        # populate response with detected classes
        # time.sleep(0.001)

        return response

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    detector_pb2_grpc.add_DetectorServicer_to_server(Detector(), server)
    server.add_insecure_port('[::]:8000')
    server.start()
    print("Server started, listening on port 8000")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()