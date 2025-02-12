'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
from openvino.inference_engine import IENetwork, IECore
import cv2
import math
import ngraph as ng

class Gaze_Estimation_Model:
    '''
    Class for the Face Detection Model.
    '''

    def __init__(self, model_name, device='CPU', extensions=None):
        self.core = None
        self.network = None
        self.input = None
        self.output = None
        self.exec_network = None
        self.device = device

        self.core = IECore()
        self.network = self.core.read_network(model=str(model_name),
                                              weights=str(os.path.splitext(model_name)[0] + ".bin"))

        self.input = next(iter(self.network.input_info))
        self.output = next(iter(self.network.outputs))

    def load_model(self):
        '''
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.exec_network = self.core.load_network(self.network, self.device)
        return self.exec_network

    def predict(self, left_eye, right_eye, head_position):
        '''
        This method is meant for running predictions on the input image.
        '''
        processed_left_eye = self.preprocess_input(left_eye)
        processed_right_eye = self.preprocess_input(right_eye)
        # inference_start_time = time.time()
        self.exec_network.start_async(request_id=0,
                                      inputs={'left_eye_image': processed_left_eye,
                                              'right_eye_image': processed_right_eye,
                                              'head_pose_angles': head_position})

        if self.exec_network.requests[0].wait(-1) == 0:
            #     inference_end_time = time.time()
            #     total_inference_time = inference_end_time - inference_start_time
            result = self.exec_network.requests[0].output_blobs[self.output].buffer
            cords = self.preprocess_output(result[0], head_position)
            return result[0], cords

    def check_model(self):
        supported_layers = self.core.query_network(network=self.network, device_name=self.device)
        function = ng.function_from_cnn(self.network).get_ops()
        unsupported_layers = [layer for layer in function if layer.friendly_name not in supported_layers]
        if len(unsupported_layers) > 0:
            print("Please check extention for these unsupported layers =>" + str(unsupported_layers))
            exit(1)
        print("All layers are supported !!")

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        net_input_shape = self.network.input_info['right_eye_image'].input_data.shape
        p_frame = cv2.resize(image, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose(2, 0, 1)
        # p_frame = np.expand_dims(p_frame, axis=1)
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame

    def preprocess_output(self, output, head_position):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''

        roll = head_position[2]
        gaze_vector = output / cv2.norm(output)

        cosValue = math.cos(roll * math.pi / 180.0)
        sinValue = math.sin(roll * math.pi / 180.0)


        x = gaze_vector[0] * cosValue * gaze_vector[1] * sinValue
        y = gaze_vector[0] * sinValue * gaze_vector[1] * cosValue
        return (x, y)
