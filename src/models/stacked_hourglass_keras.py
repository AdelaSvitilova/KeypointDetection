import torch
import torch.nn as nn
from .base_model import KerasModel

class StackedHourglassKeras(KerasModel):
    pass

# from keras.models import Model
# from keras.layers import Input, Conv2D, SeparableConv2D, BatchNormalization, Add, MaxPool2D, UpSampling2D
# from keras.optimizers import RMSprop
# from keras.losses import mean_squared_error
# import keras.backend as K
# from typing import Tuple, Callable, List

# import os
# import datetime
# from typing import Tuple, Optional
# import numpy as np
# import cv2
# from imageio import imread
# import keras
# from keras.callbacks import CSVLogger
# from keras.models import model_from_json
# from keras.optimizers import RMSprop
# from keras.losses import mean_squared_error


# def create_hourglass_network(
#     num_classes: int,
#     num_stacks: int,
#     num_channels: int,
#     inres: Tuple[int, int],
#     outres: Tuple[int, int],
#     bottleneck: Callable
# ) -> Model:
#     """Vytvoří Hourglass Network model s daným počtem stacků a kanálů."""
#     inp = Input(shape=(inres[0], inres[1], 3))
#     front_features = create_front_module(inp, num_channels, bottleneck)

#     head_next_stage = front_features
#     outputs: List = []

#     for i in range(num_stacks):
#         head_next_stage, head_to_loss = hourglass_module(head_next_stage, num_classes, num_channels, bottleneck, i)
#         outputs.append(head_to_loss)

#     model = Model(inputs=inp, outputs=outputs)
#     model.compile(optimizer=RMSprop(learning_rate=5e-4), loss=mean_squared_error, metrics=["accuracy"] * num_stacks)
#     return model


# def hourglass_module(
#     bottom,
#     num_classes: int,
#     num_channels: int,
#     bottleneck: Callable,
#     hgid: int
# ):
#     left_features = create_left_half_blocks(bottom, bottleneck, hgid, num_channels)
#     rf1 = create_right_half_blocks(left_features, bottleneck, hgid, num_channels)
#     head_next_stage, head_parts = create_heads(bottom, rf1, num_classes, hgid, num_channels)
#     return head_next_stage, head_parts


# def bottleneck_block(bottom, num_out_channels: int, block_name: str):
#     """Standardní residual block s 3 conv vrstvami."""
#     _skip = bottom if bottom.shape[-1] == num_out_channels else \
#         Conv2D(num_out_channels, (1, 1), activation='relu', padding='same', name=f"{block_name}_skip")(bottom)

#     _x = Conv2D(num_out_channels // 2, (1, 1), activation='relu', padding='same', name=f"{block_name}_conv1")(bottom)
#     _x = BatchNormalization()(_x)
#     _x = Conv2D(num_out_channels // 2, (3, 3), activation='relu', padding='same', name=f"{block_name}_conv2")(_x)
#     _x = BatchNormalization()(_x)
#     _x = Conv2D(num_out_channels, (1, 1), activation='relu', padding='same', name=f"{block_name}_conv3")(_x)
#     _x = BatchNormalization()(_x)
#     _x = Add(name=f"{block_name}_residual")([_skip, _x])
#     return _x


# def bottleneck_mobile(bottom, num_out_channels: int, block_name: str):
#     """Lightweight residual block s depthwise separable conv."""
#     _skip = bottom if K.int_shape(bottom)[-1] == num_out_channels else \
#         SeparableConv2D(num_out_channels, (1, 1), activation='relu', padding='same', name=f"{block_name}_skip")(bottom)

#     _x = SeparableConv2D(num_out_channels // 2, (1, 1), activation='relu', padding='same', name=f"{block_name}_conv1")(bottom)
#     _x = BatchNormalization()(_x)
#     _x = SeparableConv2D(num_out_channels // 2, (3, 3), activation='relu', padding='same', name=f"{block_name}_conv2")(_x)
#     _x = BatchNormalization()(_x)
#     _x = SeparableConv2D(num_out_channels, (1, 1), activation='relu', padding='same', name=f"{block_name}_conv3")(_x)
#     _x = BatchNormalization()(_x)
#     _x = Add(name=f"{block_name}_residual")([_skip, _x])
#     return _x


# def create_front_module(inp, num_channels: int, bottleneck: Callable):
#     """Front module: počáteční konvoluce a residual bloky."""
#     _x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', name='front_conv1')(inp)
#     _x = BatchNormalization()(_x)
#     _x = bottleneck(_x, num_channels // 2, 'front_res1')
#     _x = MaxPool2D((2, 2), strides=(2, 2))(_x)
#     _x = bottleneck(_x, num_channels // 2, 'front_res2')
#     _x = bottleneck(_x, num_channels, 'front_res3')
#     return _x


# def create_left_half_blocks(bottom, bottleneck: Callable, hglayer: int, num_channels: int):
#     """Vytvoří levý blok hourglass modulu."""
#     hgname = f"hg{hglayer}"
#     f1 = bottleneck(bottom, num_channels, f"{hgname}_l1")
#     _x = MaxPool2D((2, 2), strides=(2, 2))(f1)
#     f2 = bottleneck(_x, num_channels, f"{hgname}_l2")
#     _x = MaxPool2D((2, 2), strides=(2, 2))(f2)
#     f4 = bottleneck(_x, num_channels, f"{hgname}_l4")
#     _x = MaxPool2D((2, 2), strides=(2, 2))(f4)
#     f8 = bottleneck(_x, num_channels, f"{hgname}_l8")
#     return f1, f2, f4, f8


# def bottom_layer(lf8, bottleneck: Callable, hgid: int, num_channels: int):
#     lf8_connect = bottleneck(lf8, num_channels, f"{hgid}_lf8")
#     _x = bottleneck(lf8, num_channels, f"{hgid}_lf8_x1")
#     _x = bottleneck(_x, num_channels, f"{hgid}_lf8_x2")
#     _x = bottleneck(_x, num_channels, f"{hgid}_lf8_x3")
#     return Add()([_x, lf8_connect])


# def connect_left_to_right(left, right, bottleneck: Callable, name: str, num_channels: int):
#     _xleft = bottleneck(left, num_channels, f"{name}_connect")
#     _xright = UpSampling2D()(right)
#     _x = Add()([_xleft, _xright])
#     out = bottleneck(_x, num_channels, f"{name}_connect_conv")
#     return out


# def create_right_half_blocks(leftfeatures, bottleneck: Callable, hglayer: int, num_channels: int):
#     lf1, lf2, lf4, lf8 = leftfeatures
#     rf8 = bottom_layer(lf8, bottleneck, hglayer, num_channels)
#     rf4 = connect_left_to_right(lf4, rf8, bottleneck, f'hg{hglayer}_rf4', num_channels)
#     rf2 = connect_left_to_right(lf2, rf4, bottleneck, f'hg{hglayer}_rf2', num_channels)
#     rf1 = connect_left_to_right(lf1, rf2, bottleneck, f'hg{hglayer}_rf1', num_channels)
#     return rf1


# def create_heads(prelayerfeatures, rf1, num_classes: int, hgid: int, num_channels: int):
#     head = Conv2D(num_channels, (1, 1), activation='relu', padding='same', name=f'{hgid}_conv1')(rf1)
#     head = BatchNormalization()(head)
#     head_parts = Conv2D(num_classes, (1, 1), activation='linear', padding='same', name=f'{hgid}_parts')(head)
#     head = Conv2D(num_channels, (1, 1), activation='linear', padding='same', name=f'{hgid}_conv2')(head)
#     head_m = Conv2D(num_channels, (1, 1), activation='linear', padding='same', name=f'{hgid}_conv3')(head_parts)
#     head_next_stage = Add()([head, head_m, prelayerfeatures])
#     return head_next_stage, head_parts


# def euclidean_loss(y_true, y_pred):
#     return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

# class StackedHourglassKeras(BaseModel):
#     """Wrapper pro Hourglass Network s tréninkem, resume a inference."""

#     def __init__(self, num_classes: int, num_stacks: int, num_channels: int,
#                  inres: Tuple[int, int], outres: Tuple[int, int]):
#         self.num_classes = num_classes
#         self.num_stacks = num_stacks
#         self.num_channels = num_channels
#         self.inres = inres
#         self.outres = outres
#         self.model: Optional[keras.models.Model] = None

#     def build_model(self, mobile: bool = False, show: bool = False):
#         """Vytvoří Hourglass model. Volitelně mobilní verzi."""
#         bottleneck_fn = bottleneck_mobile if mobile else bottleneck_block

#         self.model = create_hourglass_network(
#             self.num_classes,
#             self.num_stacks,
#             self.num_channels,
#             self.inres,
#             self.outres,
#             bottleneck_fn
#         )

#         if show:
#             self.model.summary()

#     def train(self, batch_size: int, model_path: str, epochs: int):
#         """Trénink modelu od začátku."""
#         train_dataset = MPIIDataGen(
#             "../../data/mpii/mpii_annotations.json",
#             "../../data/mpii/images",
#             inres=self.inres,
#             outres=self.outres,
#             is_train=True
#         )
#         train_gen = train_dataset.generator(
#             batch_size, self.num_stacks, sigma=1, is_shuffle=True,
#             rot_flag=True, scale_flag=True, flip_flag=True
#         )

#         # zajisti, že složka existuje
#         os.makedirs(model_path, exist_ok=True)

#         csv_filename = os.path.join(
#             model_path,
#             "csv_train_" + datetime.datetime.now().strftime('%H-%M') + ".csv"
#         )

#         # normalizace cesty (odstraní smíchaná lomítka)
#         csv_filename = os.path.normpath(csv_filename)

#         csvlogger = CSVLogger(csv_filename)
#         checkpoint = EvalCallBack(model_path, self.inres, self.outres)

#         self.model.fit(
#             train_gen,
#             steps_per_epoch=train_dataset.get_dataset_size() // batch_size,
#             epochs=epochs,
#             callbacks=[csvlogger, checkpoint]
#         )

#     def resume_train(self, batch_size: int, model_json: str, model_weights: str,
#                      init_epoch: int, epochs: int):
#         """Pokračování tréninku z uloženého modelu a vah."""
#         self.load_model(model_json, model_weights)
#         self.model.compile(optimizer=RMSprop(lr=5e-4),
#                            loss=mean_squared_error, metrics=["accuracy"])

#         train_dataset = MPIIDataGen(
#             "../../data/mpii/mpii_annotations.json",
#             "../../data/mpii/images",
#             inres=self.inres,
#             outres=self.outres,
#             is_train=True
#         )
#         train_gen = train_dataset.generator(
#             batch_size, self.num_stacks, sigma=1, is_shuffle=True,
#             rot_flag=True, scale_flag=True, flip_flag=True
#         )

#         model_dir = os.path.dirname(os.path.abspath(model_json))
#         # zajisti, že složka existuje
#         os.makedirs(model_dir, exist_ok=True)

#         csv_filename = os.path.join(
#             model_dir,
#             "csv_train_" + datetime.datetime.now().strftime('%H-%M') + ".csv"
#         )

#         # normalizace cesty (odstraní smíchaná lomítka)
#         csv_filename = os.path.normpath(csv_filename)

#         csvlogger = CSVLogger(csv_filename)
#         checkpoint = EvalCallBack(model_dir, self.inres, self.outres)

#         self.model.fit_generator(
#             generator=train_gen,
#             steps_per_epoch=train_dataset.get_dataset_size() // batch_size,
#             initial_epoch=init_epoch,
#             epochs=epochs,
#             callbacks=[csvlogger, checkpoint]
#         )

#     def load_model(self, model_json: str, model_weights: str):
#         """Načte model z JSON a váh HDF5."""
#         with open(model_json, 'r') as f:
#             self.model = model_from_json(f.read())
#         self.model.load_weights(model_weights)

#     def inference_rgb(self, rgbdata: np.ndarray, orgshape: Tuple[int, int, int],
#                       mean: Optional[np.ndarray] = None):
#         """Predikce pro RGB obraz."""
#         scale = (orgshape[0] / self.inres[0], orgshape[1] / self.inres[1])
#         imgdata = cv2.resize(rgbdata, (self.inres[1], self.inres[0]))  # width, height

#         if mean is None:
#             mean = np.array([0.4404, 0.4440, 0.4327], dtype=np.float32)

#         imgdata = normalize(imgdata, mean)
#         input_tensor = imgdata[np.newaxis, :, :, :]
#         out = self.model.predict(input_tensor)
#         return out[-1], scale

#     def inference_file(self, imgfile: str, mean: Optional[np.ndarray] = None):
#         """Predikce přímo z obrázkového souboru."""
#         imgdata = imread(imgfile)
#         return self.inference_rgb(imgdata, imgdata.shape, mean)