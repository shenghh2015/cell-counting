import segmentation_models_v1 as sm
# from segmentation_models_v1.models.nestnet import Nestnet
import numpy as np
# model = sm.Nestnet('efficientnetb2', input_shape =(512,512,3), classes=4)
model = sm.Nestnet('efficientnetb0', input_shape =(512,512,3), classes=4)
model.summary()
y=model.predict(np.ones((1,512,512,3)))
y.shape