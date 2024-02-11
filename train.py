import sys
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.distribute.collective_all_reduce_strategy import CollectiveAllReduceExtended

CollectiveAllReduceExtended._enable_check_health = False
tf.compat.v1.disable_eager_execution()

PATH_TO_DATA: str = r"data/"
TRAIN_DATA_PATH: str = PATH_TO_DATA + "train_data"
TEST_DATA_PATH: str = PATH_TO_DATA + "test_data"
BATCH_SIZE = 16
TRAIN_SIZE = 4777
TEST_SIZE = 300

#Descriptor of file structure of a dataset (example dataset_train)
#./dataset
# ->class 1
# ->class 2
#  .
#  .

if __name__ ==  '__main__':
    # Check available GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Set the GPU to be used
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    else:
        print("No GPU available, enter any input to use CPU")
        # Set the CPU to be used
        user_input = input()
        if user_input:
            tf.config.set_visible_devices([], 'GPU')
            print("CPU is being used")
        else:
            sys.exit()

    def get_datagen(dataset, aug=False):
        if aug:
            datagen = ImageDataGenerator(
                                rescale=1./255,
                                featurewise_center=False,
                                featurewise_std_normalization=False,
                                rotation_range=25,
                                width_shift_range=0.3,
                                height_shift_range=0.3,
                                shear_range=0.5,
                                zoom_range=0.3,
                                horizontal_flip=False,
                                brightness_range=[0.8,1.1])
        else:
            datagen = ImageDataGenerator(rescale=1./255)

        return datagen.flow_from_directory(
                dataset,
                target_size=(224, 224),
                color_mode='rgb',
                shuffle = True,
                class_mode='categorical',
                batch_size=BATCH_SIZE)

    train_generator  = get_datagen(TRAIN_DATA_PATH, True)
    test_generator   = get_datagen(TEST_DATA_PATH, False)

    def load_model(m):
        if m == 'MN':
        model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape=(224,224,3), alpha=1.0, include_top=False, weights='imagenet', pooling=None)
        
        for layer in model.layers[:-1]:
            layer.trainable = False

        mobile_net = Flatten()(model.output)
        mobile_net = Dropout(0.3)(mobile_net)
        mobile_net = Dense(4096, activation='relu')(mobile_net)
        mobile_net = Dropout(0.3)(mobile_net)
        mobile_net = Dense(1024, activation='relu')(mobile_net)
        mobile_net = Dropout(0.3)(mobile_net)
        mobile_net = Dense(12, activation='softmax')(mobile_net)

        mobile_net_mobile = Model(model.input, mobile_net, name='Altered_MobileNet')
        mobile_net_mobile.summary()

        model = mobile_net_mobile

        elif m =='RN':
        pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                            input_shape=(224,224,3),
                            pooling='avg',classes=5,
                            weights='imagenet')
        for layer in pretrained_model.layers[:-1]:
                layer.trainable=False

        resnet_model = Flatten()(pretrained_model.output)
        resnet_model = Dropout(0.3)(resnet_model)
        resnet_model = Dense(4096, activation='relu')(resnet_model)
        resnet_model = Dropout(0.3)(resnet_model)
        resnet_model = Dense(1024, activation='relu')(resnet_model)
        resnet_model = Dropout(0.3)(resnet_model)
        resnet_model = Dense(12, activation='softmax')(resnet_model)

        Altered_ResNet = Model(pretrained_model.input, resnet_model, name='Altered_MobileNet')

        model = Altered_ResNet

        elif m == 'VG':
        pretrained_model = VGG16(include_top=False,
                            input_shape=(224,224,3),
                            pooling='avg',classes=5,
                            weights='imagenet')
        for layer in pretrained_model.layers[:-1]:
                layer.trainable=False

        vgg_model = Flatten()(pretrained_model.output)
        vgg_model = Dropout(0.3)(vgg_model)
        vgg_model = Dense(4096, activation='relu')(vgg_model)
        vgg_model = Dropout(0.3)(vgg_model)
        vgg_model = Dense(1024, activation='relu')(vgg_model)
        vgg_model = Dropout(0.3)(vgg_model)
        vgg_model = Dense(12, activation='softmax')(vgg_model)

        Altered_VGG = Model(pretrained_model.input, vgg_model, name='Altered_MobileNet')

        model = Altered_VGG

        elif m == 'IC':
        pretrained_model = InceptionV3(include_top=False,
                            input_shape=(224,224,3),
                            pooling='avg',classes=5,
                            weights='imagenet')
        for layer in pretrained_model.layers[:-1]:
                layer.trainable=False

        incep_model = Flatten()(pretrained_model.output)
        incep_model = Dropout(0.3)(incep_model)
        incep_model = Dense(4096, activation='relu')(incep_model)
        incep_model = Dropout(0.3)(incep_model)
        incep_model = Dense(1024, activation='relu')(incep_model)
        incep_model = Dropout(0.3)(incep_model)
        incep_model = Dense(12, activation='softmax')(incep_model)

        Altered_Incep = Model(pretrained_model.input, incep_model, name='Altered_MobileNet')

        model = Altered_Incep

        return model

    model = load_model('VG')

    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    sgd = tf.keras.optimizers.SGD(learning_rate=0.001)
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',mode='max',factor=0.5, patience=10, min_lr=0.001, verbose=1)
    early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1,
                                                mode='auto', baseline=None, restore_best_weights=True)
    model.compile(loss='categorical_crossentropy',
                    optimizer=adam, metrics=['accuracy'])

    history = model.fit(
        train_generator,
        validation_data=test_generator, 
        steps_per_epoch=TRAIN_SIZE// BATCH_SIZE,
        validation_steps=TEST_SIZE// BATCH_SIZE,
        shuffle=True,
        epochs=50,
        callbacks=[early_stopper],
        use_multiprocessing=False,
    )

    model.save("./VGG_Naruto_Model")
