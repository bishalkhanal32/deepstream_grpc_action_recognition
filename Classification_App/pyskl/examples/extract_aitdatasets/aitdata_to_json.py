import os
import json

data_json = []

baseloc = "./AIT_dataset/"
dataloc = ["fall", "unstable", "activity"]

# labels = ["bend", "sit", "sit_clap", "sit", "sit", "sit", "sit_exercise",
#              "stand", "stand_clap", "stand", "stand", "stand", "stand_exercise", "walk", "fall", "unstable"]
# labels = ["bend", "sit", "sit_clap", "sit_phonecall", "sit_point", "sit_drink", "sit_exercise", "stand", 
#             "stand_clap", "stand_phonecall", "stand_point", "stand_drink", "stand_exercise", "walk", "fall", "unstable"]

label = {"bend": 0, "sit": 1, "sit_clap": 2, "sit_phonecall": 3, "sit_point": 4, "sit_checktime": 5, "sit_wave": 6,
             "stand": 7, "stand_clap": 8, "stand_phonecall": 9, "stand_point": 10, "stand_checktime": 11, "stand_wave": 12,
             "walk": 13, "fall": 14, "unstable": 15}

for i, loc in enumerate(dataloc):
    fullloc = baseloc + loc
    for cam in os.listdir(fullloc):
        for person in os.listdir(os.path.join(fullloc, cam)):
            if i!=2:
                for video in os.listdir(os.path.join(fullloc, cam, person)):
                    info = {"vid_name": video, "label": label[loc], "label_name": loc}
                    data_json.append(info)
            else:
                for action in os.listdir(os.path.join(fullloc, cam, person)):
                    for video in os.listdir(os.path.join(fullloc, cam, person, action)):
                        if "bend" in action:
                            action = "bend"
                        try:
                            info = {"vid_name": video, "label": label[action], "label_name": action}
                            data_json.append(info)
                        except:
                            print('error! see this:')
                            print(video, action, cam, loc)
                            continue


with open("aitdata.json", "w") as outfile:
    json.dump(data_json, outfile)
                    
    

