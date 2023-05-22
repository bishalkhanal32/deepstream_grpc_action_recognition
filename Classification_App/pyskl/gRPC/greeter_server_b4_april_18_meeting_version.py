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
        self.objects = {}
        self.max_len = 90
        self.fill_len = 60
        self.ageUpdateThreshold = 120
        self.ageInferInterval = 20
        self.V = 17
        self.config = load_config("/home/bishal/Documents/thesis/pyskl/configs/posec3d/slowonly_r50_ait/joint_3d_gcn_tcn_run.py")
        self.model = load_model(config=self.config, checkpoint="/home/bishal/Documents/thesis/pyskl/gRPC/pc3d_gcn_tcn_2_2_1_aitdataset.pth")
        # self.keypoint_score = np.ones(shape = (1, self.max_len, self.V), dtype=np.float16)
        # self.keypoint = np.ones(shape = (1, self.max_len, self.V, 2), dtype=np.float16)
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

        h = 360
        w = 640
        fake_anno = dict(
            frame_dir='',
            label=-1,
            img_shape=(h, w),
            original_shape=(h, w),
            start_index=0,
            modality='Pose',
            total_frames=max_len)

        oneInferenceDone = False
        receivedObjectIds = []

        # Perform detection on the image and return results
        response = detector_pb2.ActionResult()
        # receiving and storing messages
        for detection_result in request.detectionResult:
            for detection in detection_result.detection:
                objectId = detection.objectId
                streamId = detection.streamId

                receivedObjectIds.append(objectId)
                
                if objectId not in self.pose_results_dict.keys(): 
                    self.pose_results_dict[objectId] = {"poseSeq": deque(maxlen=max_len), "ageUpdate": 0, "ageInfer": 0, "filled": False, "streamId": streamId}

                joint_points = [] # shape V x C
                for i, point in enumerate(detection.points):
                    if i!=17: # skipping the neck (18 keypoints from trtpose, my model needs 17 keypoints)
                        joint_points.append([point.x, point.y])
                
                self.pose_results_dict[objectId]["poseSeq"].append(joint_points) # shape T x V x 2
                self.pose_results_dict[objectId]["ageUpdate"] = 0

                if len(self.pose_results_dict[objectId]["poseSeq"]) >= self.fill_len:
                    print("Ready: ", objectId)
                    self.pose_results_dict[objectId]["filled"] = True
                    self.pose_results_dict[objectId]["ageInfer"] += 1 # the inferred objectId will be reset to 0, so no need to worry


        # performing inference
        max_age = -1
        max_age_objectId = None

        for ID in receivedObjectIds:
            if self.pose_results_dict[ID]["filled"]:
                age_infer = self.pose_results_dict[ID]["ageInfer"]
                if age_infer > max_age:
                    max_age = age_infer
                    max_age_objectId = ID

        classId = -1
        if max_age_objectId is not None:
            # do inference
            print("infering: ", max_age_objectId)
            keypoint = np.array(self.pose_results_dict[max_age_objectId]["poseSeq"], dtype=np.float16) # T x V x 2
            keypoint = np.expand_dims(keypoint, axis=0) # 1 x T x V x 2
            seqLen = keypoint.shape[1]
            fake_anno['keypoint'] = keypoint
            fake_anno['keypoint_score'] = np.ones(shape = (1, seqLen, self.V), dtype=np.float16)
            fake_anno['total_frames'] = seqLen
            scores = inference_(model = self.model,  data = fake_anno) # error is here

            print(scores)
            action_label = self.label_map[scores[0][0]]
            classId = scores[0][0]
            print(action_label)

            self.pose_results_dict[max_age_objectId]["ageInfer"] = 0

            old_pose_estimation = self.pose_results_dict[max_age_objectId]["poseSeq"].popleft()
            self.pose_results_dict[max_age_objectId]["filled"] = False

            action_class = detector_pb2.ActionClass()
            action_class.objectId = max_age_objectId
            action_class.streamId = self.pose_results_dict[max_age_objectId]["streamId"]
            action_class.classId = classId
            response.classes.append(action_class)

        else:
            # just return response of -1
            print("No objectId has filled=True.")
            action_class = detector_pb2.ActionClass()
            action_class.objectId = -1
            action_class.streamId = -1
            action_class.classId = -1 
            response.classes.append(action_class)

        # removing and updating database
        pose_results_dict_copy = self.pose_results_dict.copy()
        for objectId, obj_data in pose_results_dict_copy.items():
            if objectId not in receivedObjectIds:
                self.pose_results_dict[objectId]["ageUpdate"] += 1
            # else:
            #     self.pose_results_dict[objectId]["ageUpdate"] = 0

            # if (objectId != max_age_objectId) and obj_data["filled"]:
            #     self.pose_results_dict[objectId]["ageInfer"] += 1
            
            if obj_data["ageUpdate"] > self.ageUpdateThreshold:
                print("deleting: ", objectId)
                del self.pose_results_dict[objectId]

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