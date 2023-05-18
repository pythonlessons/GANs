import keras
import numpy as np
import cv2

model = keras.models.load_model("Models/02_WGANGP_faces/model/generator.h5", compile=False)

while True:
    noise = np.random.normal(0, 1, (1, model.input_shape[1]))
    generated_image = model.predict(noise, verbose=False) * 127.5 + 127.5
    generated_image = generated_image[..., ::-1].astype(np.uint8)
    cv2.imshow("Generated Image", generated_image[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()