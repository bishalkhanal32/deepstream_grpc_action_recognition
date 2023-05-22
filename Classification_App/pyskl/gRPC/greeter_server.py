import grpc
from concurrent import futures
import time

import detector_pb2
import detector_pb2_grpc

import numpy as np
from collections import deque
from inference import load_config, load_model, inference_

import csv
def append_data(src_id, frame_num, obj_id, act_id, act_name):
    with open('data.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write a new row with the data
        writer.writerow([src_id, frame_num, obj_id, act_id, act_name])

class Detector(detector_pb2_grpc.DetectorServicer):

    def __init__(self):
        # self.model = None
        self.pose_results_dict = {}
        self.objects = {}
        self.max_len = 90
        self.fill_len = 70
        self.ageUpdateThreshold = 120
        self.ageInferInterval = 10
        self.numberOfPosesToRemove = 10
        self.V = 17
        self.config = load_config("./../configs/posec3d/slowonly_r50_ait/joint_3d_gcn_tcn_run.py")
        self.model = load_model(config=self.config, checkpoint="./pc3d_gcn_tcn_2_2_1_aitdataset.pth")
        # self.keypoint_score = np.ones(shape = (1, self.max_len, self.V), dtype=np.float16)
        # self.keypoint = np.ones(shape = (1, self.max_len, self.V, 2), dtype=np.float16)
        self.label_map = [x.strip() for x in open("./aitdata_labels.txt").readlines()]

        # Create a new file to store the data
        with open('data.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header row
            writer.writerow(['SourceID', 'FrameNum', 'ObjectID', 'ActionID', 'ActionName'])



    def Detect(self, request, context):
        print("Received detection request")
        batch = []
        objectIdsInBatch = []
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

        try:
            # receiving and storing messages
            for detection_result in request.detectionResult:
                sourceId = detection_result.sourceId
                frameNum = detection_result.frameNum
                for detection in detection_result.detection:
                    if len(detection.points) >= 17: # sometime there is no pose points coming from yolo->ViTPose
                        objectId = detection.objectId

                        receivedObjectIds.append(objectId)
                        
                        if objectId not in self.pose_results_dict.keys():
                            self.pose_results_dict[objectId] = {"poseSeq": deque(maxlen=max_len), "ageUpdate": 0, "ageInfer": 0, "filled": False, "sourceId": sourceId, "frameNum": frameNum}

                        joint_points = [] # shape V x C

                        # if len(detection.points) < 17:
                        #     if len(detection.points) == 0:
                        #         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!what the heck!!!!!!!!!!!")
                        #         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!what the heck!!!!!!!!!!!")
                        #         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!what the heck!!!!!!!!!!!")
                        #         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!what the heck!!!!!!!!!!!")
                        #     exit()
                        for i, point in enumerate(detection.points):
                            if i!=17: # skipping the neck (18 keypoints from trtpose, my model needs 17 keypoints)
                                joint_points.append([point.x, point.y])
                        # print("")
                        
                        self.pose_results_dict[objectId]["poseSeq"].append(joint_points) # shape T x V x 2
                        self.pose_results_dict[objectId]["ageUpdate"] = 0
                        self.pose_results_dict[objectId]["frameNum"] = frameNum

                        if len(self.pose_results_dict[objectId]["poseSeq"]) >= self.fill_len:
                            print("Ready: ", objectId)
                            self.pose_results_dict[objectId]["filled"] = True
                            self.pose_results_dict[objectId]["ageInfer"] += 1 # the inferred objectId will be reset to 0, so no need to worry
        except:
            print("---------------------------Error-1------------------------------")
            action_class = detector_pb2.ActionClass()
            action_class.objectId = -1
            action_class.sourceId = -1
            action_class.classId = -1 
            response.classes.append(action_class)

        
        # performing inference
        objectIdToInfer = None

        for ID in receivedObjectIds:
            if self.pose_results_dict[ID]["filled"]:
                age_infer = self.pose_results_dict[ID]["ageInfer"]
                if age_infer > self.ageInferInterval:
                    objectIdToInfer = ID
        
        try:
            classId = -1
            if objectIdToInfer is not None:
                # do inference
                print("infering: ", objectIdToInfer)
                try:
                    keypoint = np.array(self.pose_results_dict[objectIdToInfer]["poseSeq"], dtype=np.float16) # T x V x 2
                except:
                    print("---------------------Error-2aaaaaaaaaaa-----------------------------")
                    print("object id: ", objectIdToInfer)
                    print("Pose results: ", self.pose_results_dict[objectIdToInfer])
                    print("object id: ", objectIdToInfer)
                    # print("QUEUE SIZE: ", self.pose_results_dict[objectIdToInfer]["poseSeq"].qsize())
                    action_class = detector_pb2.ActionClass()
                    action_class.objectId = -1
                    action_class.sourceId = -1
                    action_class.classId = -1 
                    response.classes.append(action_class)

                keypoint = np.expand_dims(keypoint, axis=0) # 1 x T x V x 2
                seqLen = keypoint.shape[1]
                fake_anno['keypoint'] = keypoint
                fake_anno['keypoint_score'] = np.ones(shape = (1, seqLen, self.V), dtype=np.float16)
                fake_anno['total_frames'] = seqLen
                scores = inference_(model = self.model,  data = fake_anno)
                
                print(scores)
                action_label = self.label_map[scores[0][0]]
                classId = scores[0][0]
                print(action_label)

                self.pose_results_dict[objectIdToInfer]["ageInfer"] = 0
                
                for i in range(self.numberOfPosesToRemove):
                    old_pose_estimation = self.pose_results_dict[objectIdToInfer]["poseSeq"].popleft()

                self.pose_results_dict[objectIdToInfer]["filled"] = False

                action_class = detector_pb2.ActionClass()
                action_class.objectId = objectIdToInfer
                action_class.sourceId = self.pose_results_dict[objectIdToInfer]["sourceId"]
                action_class.classId = classId
                response.classes.append(action_class)

                append_data(action_class.sourceId , self.pose_results_dict[objectIdToInfer]["frameNum"], objectIdToInfer, classId, action_label)

            else:
                # just return response of -1
                print("No objectId has filled=True.")
                action_class = detector_pb2.ActionClass()
                action_class.objectId = -1
                action_class.sourceId = -1
                action_class.classId = -1 
                response.classes.append(action_class)
        except:
            print("---------------------Error-2-----------------------------")
            action_class = detector_pb2.ActionClass()
            action_class.objectId = -1
            action_class.sourceId = -1
            action_class.classId = -1 
            response.classes.append(action_class)
        
        try:
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
        
        except:
            print("----------------------Error-3---------------------------")
            action_class = detector_pb2.ActionClass()
            action_class.objectId = -1
            action_class.sourceId = -1
            action_class.classId = -1 
            response.classes.append(action_class)

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
