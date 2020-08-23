#
# Copyright Amazon AWS DeepLens, 2017
#

import os
import greengrasssdk
from threading import Timer
import time
import awscam
import mo
import cv2
from threading import Thread

# Creating a greengrass core sdk client
client = greengrasssdk.client('iot-data')

# The information exchanged between IoT and clould has
# a topic and a message body.
# This is the topic that this code uses to send messages to cloud
iotTopic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])
fem_image_name = "Advert-Femme.png"
hom_image_name = "Advert-Homme.png"
fem_image = cv2.imread(fem_image_name)
hom_image = cv2.imread(hom_image_name)
resized_fem_image = cv2.resize(fem_image, (0, 0), fx=0.1, fy=0.1)
ad_mod = True
output_written = False
deep_ads_blue = (246, 87, 35)
recognize_face_counter = 0

ret, frame = awscam.getLastFrame()
ret,jpeg = cv2.imencode('.jpg', frame)
Write_To_FIFO = True


modelPath = "/opt/awscam/artifacts/mxnet_deploy_ssd_FP16_FUSED.xml"
modelType = "ssd"
input_width = 300
input_height = 300
prob_thresh = 0.25
faceModelPath = "/opt/awscam/artifacts/mxnet_image-classification3_FP16_FUSED.xml"
face_prob_thresh = 0.80


#aux={'--output--dir': '/home/aws_cam/gender_model', '--models--dir': '/home/aws_cam/gender_model'}
#ret, face_model_path = mo.optimize('image-classification', 224, 224, "mxNet", aux)
# Load model to GPU (use {"GPU": 0} for CPU)
#error, face_model_path = mo.optimize(model_path, input_width, input_height)
mcfg = {"GPU": 1}
model = awscam.Model(modelPath, mcfg)
faceModel = awscam.Model(faceModelPath, mcfg)
client.publish(topic=iotTopic, payload="Model loaded")
gender_output_written = False

class FIFO_Thread(Thread):
    def __init__(self):
        ''' Constructor. '''
        Thread.__init__(self)

    def run(self):
        fifo_path = "/tmp/results.mjpeg"
        if not os.path.exists(fifo_path):
            os.mkfifo(fifo_path)
        f = open(fifo_path,'w')
        client.publish(topic=iotTopic, payload="Opened Pipe")
        while Write_To_FIFO:
            try:
                f.write(jpeg.tobytes())
            except IOError as e:
                continue

om_labels = ["Female", "Male"]

def recognize_face(model, frame, thresh):
    global recognize_face_counter
    recognize_face_counter += 1
    debug_output = open('/tmp/new_gender_debug_output.txt','a')
    debug_output.write('\n recognize_face called with frame: {}\n'.format(frame.shape))
    height, width, channel = frame.shape
    if (height <= 0) or (width <= 0):
        return False, -1, 0.0
    try:
        client.publish(topic=iotTopic, payload="Recognizing Outwarians")
        modelType = 'classification'
        payload = cv2.resize(frame, (input_width, input_height), interpolation=cv2.INTER_CUBIC)
        inferOutput = model.doInference(payload)
        results = model.parseResult(modelType, inferOutput)[modelType]
	prob = 0.0
        label = -1
	debug_output.write('\niteration count: {}\n'.format(recognize_face_counter))
	debug_output.write('\ngender results for iteration: {}\n'.format(results))
	male_result = results[0]
	female_result = results[1]
	debug_output.close()
	if female_result['prob'] > 0.2:
	    return True, 0, female_result['prob']
	else:
	    return True, 1, male_result['prob']
	
	#cv2.putText(frame, 'gender results: {}\n'.format(results), (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 120), 4)
        #for result in results:
         #   result_prob = result['prob']
          #  if result_prob < thresh:
           #     continue
            #if result_prob > prob:
                #label = result['label']
                #prob = result_prob
	
	#if (prob > 0.0) and (label >= 0):
	 #   return True, label, prob        
       	#else:
         #   return False, -1, 0.0
    except Exception as e:
        msg = "Test failed " + str(e)
	debug_output.write('\nexception caught trying to do model inference\n' + msg)
	debug_output.close()
        client.publish(topic=iotTopic, payload=msg)
    	return False, -1, 0.0

def greengrass_infinite_infer_run():
    try:
        
        results_thread = FIFO_Thread()
        results_thread.start()	

        # Send a starting message to IoT console
        client.publish(topic=iotTopic, payload="Face detection starts now")

        
        ret, frame = awscam.getLastFrame()
        if ret == False:
            raise Exception("Failed to get frame from the stream")
        yscale = float(frame.shape[0]/input_height)
        xscale = float(frame.shape[1]/input_width)
        
        global jpeg
        process_this_frame = False
        doInfer = True
        face_count = 0
        while doInfer:
            # Get a frame from the video stream
            ret, frame = awscam.getLastFrame()
           
            process_this_frame = not process_this_frame
            if process_this_frame == False:
                count_label = 'total faces: {}'.format(face_count)
                cv2.putText(frame, count_label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 120), 4)

                frame = cv2.resize(frame, (1344, 760))
                jpeg = fem_image
                #ret,jpeg = cv2.imencode('.jpg', frame)
                
                client.publish(topic=iotTopic, payload="Not processing this frame")
                continue

            # Raise an exception if failing to get a frame
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
            label = '{'
            for obj in parsed_results:
                if obj['prob'] < prob_thresh:
                    continue
                xmin = int( xscale * obj['xmin'] ) + int((obj['xmin'] - input_width/2) + input_width/2)
                ymin = int( yscale * obj['ymin'] )
                xmax = int( xscale * obj['xmax'] ) + int((obj['xmax'] - input_width/2) + input_width/2)
                ymax = int( yscale * obj['ymax'] )
                dx = (xmax - xmin) / 2
                dy = (ymax - ymin) / 2
                cv2.rectangle(frame, (xmin - dx, ymin - dy), (xmax + dx, ymax + dy), deep_ads_blue, 4)
                roi = frame[ymin:ymax, xmin:xmax]
                found, om_label_index, om_prob  = recognize_face(faceModel, roi, face_prob_thresh)
                if found == True:
                    
		    if om_label_index == 1:
			male_count += 1
			person_array.append(1)
		    else:
			female_count += 1
			person_array.append(0)
		    face_count = face_count + 1
                    label += '"{}": {:.2f}, '.format(str(om_labels[om_label_index]), om_prob)
                    label_show = '#{}'.format(face_count)
                    #label_show = '{}: {:.2f}, #{}'.format(str(om_labels[om_label_index]), om_prob, face_count)                   
                else:
                    label += '"{}": {:.2f}, '.format(str(obj['label']), obj['prob'])
                    label_show = '{}: {:.2f}'.format(str(obj['label']), obj['prob'])
                    
                cv2.putText(frame, label_show, (xmin - dx, ymin - dy - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, deep_ads_blue, 4)
            label += '"null": 0.0'
            label += '}'
            count_label = 'Total Faces: {}'.format(face_count)
	    male_label = 'Male: {}, Female: {}'.format(male_count,female_count)
	    female_label = ''.format(female_count)
	    ad_label = 'Ad type: '
	    if male_count > female_count:
		ad_label += "Male"
	    elif female_count > male_count:
		ad_label += "Female"
	    elif female_count >= 1 and male_count >= 1 and face_count != 0:
		ad_label += "Group"
	    x_orig = 30
	    y_orig = frame.shape[0] - 350
	    x_dest = x_orig + int((frame.shape[1]/2.5))
	    y_dest = y_orig + 250
	    overlay = frame.copy()
	    output = frame.copy()
	    cv2.rectangle(overlay, (x_orig-20,y_orig-30), (x_dest + 5,y_dest + 5), (255,255,255),-1)            
	    cv2.putText(overlay, count_label, (x_orig, y_orig+20), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (246, 87, 35), 4)
	    cv2.putText(overlay, male_label, (x_orig, y_orig+75), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (246, 87, 35), 4)
	    cv2.putText(overlay, ad_label, (x_dest-315, y_orig+20), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (50, 50, 235), 4)
	    face_counter = 0
	    debug_output = open('/tmp/robert_debug_output.txt','a')
	    for i in range(0,face_count):
		gender = 'Male' if person_array[i] == 1 else 'Female'
		person_label = 'Person #{}: {}, Age: ?, Glasses: ?'.format(i + 1, gender)
		debug_output.write('\nperson_label: {}\n'.format(person_label))
		cv2.putText(overlay, person_label, (x_orig, y_orig+125+(60*i)), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (246, 87, 35), 4)
	    cv2.addWeighted(overlay, 0.8, output, 0.2, 0, output)
	    x_offset = frame.shape[1]- resized_fem_image.shape[1]
	    y_offset = frame.shape[0]- resized_fem_image.shape[0]
	    #debug_output = open('/tmp/robert_debug_output.txt','a')
	    #debug_output.write("\nx_offset: {}\n".format(x_offset))
	    #debug_output.write("\ny_offset: {}\n".format(y_offset))
	    #debug_output.write("\nframe.shape: {}\n".format(frame.shape))
	    #debug_output.write("\nresized_fem_image.shape: {}\n".format(resized_fem_image.shape))
	    #debug_output.close()
	    
	    #`x = 0
	    #y = 0
	    #for i in range(y_offset, frame.shape[0]):
		#y += 1
		#for j in range(x_offset, frame.shape[1]):
		    #x += 1
		    #frame[i, j] = resized_fem_image[y, x] 

            client.publish(topic=iotTopic, payload = label)
            frame = cv2.resize(output, (1344, 760))
            ret,jpeg = cv2.imencode('.jpg', frame)
            
    except Exception as e:
        msg = "Test failed: " + str(e)
        client.publish(topic=iotTopic, payload=msg)

    # Asynchronously schedule this function to be run again in 15 seconds
    Timer(5, greengrass_infinite_infer_run).start()


# Execute the function above
greengrass_infinite_infer_run()


# This is a dummy handler and will not be invoked
# Instead the code above will be executed in an infinite loop for our example
def function_handler(event, context):
    return