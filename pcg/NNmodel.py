from pcg.imports_n_settings import *
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam


def create_model2(optimizer=RMSprop,lr=0.0001, loss= 'binary_crossentropy', input_dim= 21,
                 layer_a=30, layer_b=30,layer_c=12,
                 activation_a = 'relu' ,activation_b ='relu',activation_c ='sigmoid',):

    # create model
    model = Sequential()
    model.add(Dense(layer_a, input_dim=input_dim, activation=activation_a))
    model.add(Dense(layer_b, activation=activation_b))
    model.add(Dense(layer_c, activation=activation_c))
    model.add(Dense(2, activation='softmax'))
    # Compile model
    #optimzer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=loss, optimizer=optimizer(lr=lr), metrics=['accuracy'])
    return model