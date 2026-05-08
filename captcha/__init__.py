import logging
from mltu.utils.text_utils import ctc_decoder, get_cer
from PIL import Image
import io
import json
import pandas as pd
import base64
import azure.functions as func

from . import predict
# class ImageToWordModel(OnnxInferenceModel):
#     def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.char_list = char_list

#     def predict(self, image: np.ndarray):
#         image = cv2.resize(image, self.input_shape[:2][::-1])

#         image_pred = np.expand_dims(image, axis=0).astype(np.float32)

#         preds = self.model.run(None, {self.input_name: image_pred})[0]

#         text = ctc_decoder(preds, self.char_list)[0]

#         return text

# configs = BaseModelConfigs.load("udyamcaptcha/configs.yaml")

# model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

# IMAGE_PATH = "static\temp\image.png"
captcha_type = None
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    route = req.route_params.get('route')
    valid_routes = ["a", "b", "c", "d", "e"]

    if route in valid_routes:
        captcha_type = route
        file_value = req.params.get('file')
        if not file_value:
            try:
                req_body = req.get_json()
            except ValueError:
                pass
            else:
                file_value = req_body.get('file')
                if file_value is None:
                    return func.HttpResponse("File input is required") 
    else:
        return func.HttpResponse("Please provide a valid endpoint", status_code=400)
    
    if 'base64' in file_value:
        base64str = file_value.split('base64,')[1]
        byte_data = base64.b64decode(base64str)
    else:
        img_buf=json.loads(file_value)
        byte_data = bytes([img_buf[key] for key in sorted(img_buf.keys(), key=int)])

    image_object = Image.open(io.BytesIO(byte_data))
    pred_new = predict.predict_image(image_object, captcha_type)
    print("Captcha =", pred_new)
    return func.HttpResponse(json.dumps({"Captcha": pred_new}))
