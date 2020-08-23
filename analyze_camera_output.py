import awscam
import cv2
import os
import time
from PIL import Image

modelPath = "/opt/awscam/artifacts/mxnet_deploy_ssd_FP16_FUSED.xml"
modelType = "ssd"
input_width = 300
input_height = 300
prob_thresh = 0.25
faceModelPath = "/opt/awscam/artifacts/mxnet_image-classification3_FP16_FUSED.xml"
face_prob_thresh = 0.80
face_count = 0

fem_image_name = "Advert-Femme.png"
hom_image_name = "Advert-Homme.png"
fem_image = cv2.imread(fem_image_name)
hom_image = cv2.imread(hom_image_name)
pil_fem_image = Image.open(fem_image_name)
pil_hom_image = Image.open(hom_image_name)

flag = True
mcfg = {"GPU": 1}
model = awscam.Model(modelPath, mcfg)
faceModel = awscam.Model(faceModelPath, mcfg)
last_image = 0

def recognize_face(model, frame, thresh):
    height, width, channel = frame.shape
    if (height <= 0) or (width <= 0):
        return False, -1, 0.0
    try:
        modelType = 'classification'
        payload = cv2.resize(frame, (input_width, input_height), interpolation=cv2.INTER_CUBIC)
        inferOutput = model.doInference(payload)
        results = model.parseResult(modelType, inferOutput)[modelType]
        male_result = results[0]
        female_result = results[1]
        if female_result['prob'] > 0.2:
            return True, 0, female_result['prob']
        else:
            return True, 1, male_result['prob']

    except Exception as e:
        # msg = "Test failed " + str(e)
        # print('msg: {}'.format(msg)
        return False, -1, 0.0


def analyze_frame():
    print('analyze_frame called')
    ret, frame = awscam.getLastFrame()
    if ret == False:
        raise Exception("Failed to get frame from the stream")
    yscale = float(frame.shape[0] / input_height)
    xscale = float(frame.shape[1] / input_width)
    ret, jpeg = cv2.imencode('.jpg', frame)
    if ret == False:
        raise Exception("Failed to get frame from the stream")

    # Resize frame to fit model input requirement
    frameResize = cv2.resize(frame, (input_width, input_height))

    # Run model inference on the resized frame
    inferOutput = model.doInference(frameResize)

    # Output inference result to the fifo file so it can be viewed with mplayer
    parsed_results = model.parseResult(modelType, inferOutput)['ssd']
    face_count = 0
    male_count = 0
    female_count = 0
    person_array = []
    for obj in parsed_results:
        if obj['prob'] < prob_thresh:
            continue
        xmin = int(xscale * obj['xmin']) + int((obj['xmin'] - input_width / 2) + input_width / 2)
        ymin = int(yscale * obj['ymin'])
        xmax = int(xscale * obj['xmax']) + int((obj['xmax'] - input_width / 2) + input_width / 2)
        ymax = int(yscale * obj['ymax'])
        roi = frame[ymin:ymax, xmin:xmax]
        found, om_label_index, om_prob = recognize_face(faceModel, roi, face_prob_thresh)
        if found == True:
            if om_label_index == 1:
                male_count += 1
                person_array.append(1)
            else:
                female_count += 1
                person_array.append(0)
            face_count += 1


    global last_image
    if face_count >= 1 and female_count == 0:
        if last_image != 1:
            pil_hom_image.show()
            last_image = 1

    elif face_count >= 1 and male_count == 0:
        if last_image != 2:
            pil_fem_image.show()
            last_image = 2

    elif male_count > 1 and female_count > 1:
        face_count >= 1 and female_count == 0


while flag == True:
    analyze_frame()
    time.sleep(0.5)