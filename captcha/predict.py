from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer
from mltu.configs import BaseModelConfigs
import typing
import numpy as np
import cv2
import os
import tempfile


class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shape[:2][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text



def predict_image(image_object, captcha_type):
    
    # Create a temporary file with a specific name
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        image_object.save(temp_file.name)
    # Write the image data to the temporary file
    
    # indir="udyamcaptcha/temp"
    # filename = "image.png"
    # output_path = "tempo"
    # Convert the PIL Image object to a NumPy array (compatible with OpenCV)
    img = cv2.imread(temp_file.name, cv2.IMREAD_UNCHANGED)
    if captcha_type=='e':
        cv2.imwrite(temp_file.name, img)
    elif captcha_type=='c':
        resized_image = cv2.resize(img, (300, 60))
        cv2.imwrite(temp_file.name, resized_image)
    else:
        grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        # resized_image = cv2.resize(grayscale_image, (150, 60))
        cv2.imwrite(temp_file.name, grayscale_image)

    # predict_img_path = os.path.join(indir, filename)
    # print(predict_img_path)
    # img = cv2.imread(predict_img_path, cv2.IMREAD_UNCHANGED)
    # grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    # # Resize the image to the desired dimensions
    # resized_image = cv2.resize(grayscale_image, (150, 60))  
  
    # output_path = os.path.join(indir, filename)
    # # cv2.imwrite(output_path, resized_image)
    # cv2.imwrite(predict_img_path, resized_image)
    # new_image = cv2.imread(filename)
        # Save the crop as a base64-encoded string
    captcha_types = {
    "a": "captcha/type0/configs.yaml",
    "b": "captcha/type1/configs.yaml",
    "c": "captcha/type2/configs.yaml",
    "d": "captcha/type3/configs.yaml",
    "e": "captcha/type4/configs.yaml"
}

    if captcha_type in captcha_types:
        configs = BaseModelConfigs.load(captcha_types[captcha_type])
        model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)
    else:
        raise ValueError('Captcha type not supported')

    new_image = cv2.imread(temp_file.name)
    pred_new = model.predict(new_image)
    # Delete the file
    os.remove(temp_file.name)
    return pred_new