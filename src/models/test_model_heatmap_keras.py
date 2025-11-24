from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ReLU
from .base_model import KerasModel  # abstraktní třída

class TestModelHeatmapKeras(KerasModel):
    """
    Plně konvoluční model pro predikci heatmap keypointů.
    Vstup: [batch_size, H, W, input_channels]
    Výstup: [batch_size, H, W, num_keypoints]
    """
    def __init__(self, input_channels=1, num_keypoints=5, H=256, W=256):
        super().__init__()
        self._input_channels = input_channels
        self._num_keypoints = num_keypoints
        self._H = H
        self._W = W

        # Vytvoříme model vrstvy po vrstvě
        self.inputs_layer = Input(shape=(H, W, input_channels))

        # Backbone
        x = Conv2D(16, kernel_size=3, padding='same')(self.inputs_layer)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=2)(x)  # 256 -> 128

        x = Conv2D(32, kernel_size=3, padding='same')(x)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=2)(x)  # 128 -> 64

        x = Conv2D(64, kernel_size=3, padding='same')(x)
        x = ReLU()(x)

        # Head: 1x1 konvoluce na num_keypoints kanálů
        x = Conv2D(num_keypoints, kernel_size=1, padding='same')(x)

        # Upsample zpět na původní rozlišení
        x = UpSampling2D(size=4, interpolation='bilinear')(x)  # 64 -> 256

        # Keras Model
        self.model = Model(inputs=self.inputs_layer, outputs=x)

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)
