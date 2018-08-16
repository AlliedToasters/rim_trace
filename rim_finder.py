import tensorflow as tf
from keras.layers import Input, BatchNormalization, Conv2D, Conv2DTranspose, Dense
from keras.layers import MaxPooling2D, Concatenate, Reshape, AveragePooling2D
from keras.models import Model


def make_deepnet(dim=224, w=1):
    
    def conv_layer(n, w):
        """Standard 3x3 conv layer used in model."""
        return Conv2D(filters=n*w, kernel_size=(3,3), padding='same', activation='relu')
    
    def deconv_layer(n, w):
        """Standard 'deconvolution' layer used in model."""
        return Conv2DTranspose(filters=n*w, kernel_size=(2, 2), strides=2, activation='relu')
    
    inp = Input((224, 224, 3))
    x = BatchNormalization()(inp)
    x = conv_layer(2, w)(x)
    x = conv_layer(4, w)(x)
    mid1 = BatchNormalization()(x)
    x = MaxPooling2D()(mid1)
    x = conv_layer(4, w)(x)
    x = conv_layer(8, w)(x)
    mid2 = BatchNormalization()(x)
    x = MaxPooling2D()(mid2)
    x = conv_layer(8, w)(x)
    x = conv_layer(16, w)(x)
    mid3 = BatchNormalization()(x)
    x = MaxPooling2D()(mid3)
    x = conv_layer(16, w)(x)
    x = conv_layer(32, w)(x)
    x = conv_layer(32, w)(x)
    mid4 = BatchNormalization()(x)
    x = MaxPooling2D()(mid4)
    x = conv_layer(32, w)(x)
    x = conv_layer(64, w)(x)
    mid5 = conv_layer(64, w)(x)
    x = Conv2D(filters=128*w, kernel_size=(1,1), padding='same', activation='relu')(mid5)
    x = AveragePooling2D(pool_size=(dim//16,dim//16))(x)
    x = Reshape((-1,))(x)
    x = Dense(128*w, activation='relu')(x)
    x = Dense(64*w, activation='relu')(x)
    x = Dense(196*w, activation='relu')(x)
    x = Reshape((dim//16, dim//16, -1))(x)
    x = Concatenate()([x, mid5])
    x = Conv2D(filters=64*w, kernel_size=(1,1), padding='same', activation='relu')(x)
    x = conv_layer(64, w)(x)
    x = conv_layer(64, w)(x)
    x = deconv_layer(32, w)(x)
    x = Concatenate()([mid4, x])
    x = conv_layer(32, w)(x)
    x = conv_layer(32, w)(x)
    x = deconv_layer(16, w)(x)
    x = Concatenate()([mid3, x])
    x = conv_layer(16, w)(x)
    x = conv_layer(16, w)(x)
    x = deconv_layer(8, w)(x)
    x = Concatenate()([mid2, x])
    x = conv_layer(8, w)(x)
    x = conv_layer(8, w)(x)
    x = deconv_layer(4, w)(x)
    x = Concatenate()([mid1, x])
    x = conv_layer(4, w)(x)
    x = conv_layer(4, w)(x)
    x = Conv2D(filters=1, kernel_size=(1,1), padding='same', activation='relu')(x)
    out = Reshape((224, 224))(x)
    
    model = Model(inputs=inp, outputs=out)
    
    return model

def build_model(weightpath = './models/weights.h5'):
    model = make_deepnet(w=8)
    model.load_weights(weightpath)
    return model

model = build_model()