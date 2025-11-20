from keras import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape
from keras.layers import Input
from .base_model import KerasModel


class TestModelKeras(KerasModel):
    def __init__(self, input_channels=1, num_keypoints=5, input_height=256, input_width=256):
        super().__init__()

        self._input_channels = input_channels
        self._num_keypoints = num_keypoints
        self._input_shape = (input_height, input_width, input_channels)

        # --- definice vrstev ---
        self.conv1 = Conv2D(16, kernel_size=3, padding='same', activation='relu')
        self.pool1 = MaxPooling2D(pool_size=2)

        self.conv2 = Conv2D(32, kernel_size=3, padding='same', activation='relu')
        self.pool2 = MaxPooling2D(pool_size=2)

        self.flatten = Flatten()
        self.fc = Dense(num_keypoints * 3)
        self.reshape = Reshape((num_keypoints, 3))

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.reshape(x)

    def get_model(self):
        """
        (Volitelné) vytváří Functional API model z toho subclassu.
        Užitečné pro summary().
        """
        inp = Input(shape=self._input_shape)
        out = self.call(inp)
        return Model(inp, out)
