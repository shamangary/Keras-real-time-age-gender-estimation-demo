# Keras-real-time-age-gender-estimation-demo

+ (2018/04/30 update): The demo has been combined with our latest IJCAI18 paper SSR-Net https://github.com/shamangary/SSR-Net. Plz check it out!!!

+ (2017/12/20 update):Plz use https://github.com/shamangary/Keras-MORPH2-age-estimation to train the model. The performance of small model work much better by using regression than DEX.

I use the following code to train the keras model
https://github.com/yu4u/age-gender-estimation (a very great github post)

+ Unlike the demo.py in the original github, I made several adjustments for the demo. The demo now supports both python2.7 and python3.5.

--
## 1. Small model

TYY_1stream:
```
        inputs = Input(shape=self._input_shape)

        x = Conv2D(32,(3,3),activation='relu')(inputs)
        x = MaxPooling2D(2,2)(x)
        x = Conv2D(32,(3,3),activation='relu')(x)
        x = MaxPooling2D(2,2)(x)
        x = Conv2D(64,(3,3),activation='relu')(x)
        x = MaxPooling2D(2,2)(x)
        x = Conv2D(64,(3,3),activation='relu')(x)
        x = BatchNormalization(axis=self._channel_axis)(x)

        # Classifier block
        pool = AveragePooling2D(pool_size=(4, 4), strides=(1, 1), padding="same")(x)
        flatten = Flatten()(pool)
        predictions_g = Dense(units=2, kernel_initializer=self._weight_init, use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay), activation="softmax")(flatten)
        predictions_a = Dense(units=21, kernel_initializer=self._weight_init, use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay), activation="softmax")(flatten)

        model = Model(inputs=inputs, outputs=[predictions_g, predictions_a])
```


## 2. Demo (Use "moviepy" instead of "cv2" for the frame of the video!!!)

There are a lot of issues of using cv2.VideoCapture()
https://github.com/ContinuumIO/anaconda-issues/issues/121

**Anaconda will not install opencv3 with ffmpeg properly!**

I have tried skvideo.io and pyav and something else...

[Replacement for cv2.VideoCapture] Using anaconda to install moviepy is the best option for python3.5/python2.7 with opencv3.1 or opencv3.2.
--
```
conda install -c conda-forge moviepy
conda install -c cogsci pygame
```
pygame is for showing the image with moviepy.

## 3. How to run?
1. Put you video into the folder

2. GPU with tensorflow backend (video name is the last term)
```
KERAS_BACKEND=tensorflow python TYY_demo.py mewtwo.mp4
```
3. CPU with tensorflow backend (video name is the last term)
```
KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES='' python TYY_demo.py mewtwo.mp4
```
### Python version for different display options
Since there are some problems in python2.7 for using cv2.imshow(), I set an option for python version choices. 

4. CPU with tensorflow backend with python2.7 or python3.5 (using img_clip.show())
```
KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES='' python TYY_demo.py mewtwo.mp4 '2'
```
5. CPU with tensorflow backend with python3.5 (using cv2.imshow())
```
KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES='' python TYY_demo.py mewtwo.mp4 '3'
```


## 4. Dependencies
1. Same as https://github.com/yu4u/age-gender-estimation
2. moviepy
3. pygame

## 5. Dependencies install guide (in Chinese)
http://shamangary.logdown.com/posts/3009851
