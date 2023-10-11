# DL-Image-Classification
Classification of apples and oranges using DNN and CNN models

The Apple2Orange dataset consists of 1261 photos representing apples and 1267 photos representing oranges. Both classes are divided into training and testing subsets.

The programming environment used is Python, the Tensorflow framework including the default Keras libraries.

The objective of the project is to make a comparison between two different classifiers, using neural networks and convolutional networks for the Apple2Orange dataset available at the following source:
https://www.kaggle.com/datasets/balraj98/apple2orange-dataset?resource=download

Steps:

Libraries used: numpy, pandas, matplotlib, tensorflow,

The general parameters of the images are: 30 epochs, image height 80, width 80 and batch size 80.

The training, validation and test sets were generated with the image_dataset_from_directory methods

The network structure has been created:
- For CNN
  - The preprocessing was done with a Rescaling layer through which I resized the image
  - A convolution layer and a grouping layer were added
  - A Flatten layer was added to convert the multidimensional matrix into a unidimensional one.
  - Two dense layers were added to perform the classification, the last of which has two outputs, i.e. the number of classes (apples and oranges)
- For DNN
  - The preprocessing was done with a Rescaling layer through which I resized the image
  - A Flatten layer was used to convert the multidimensional image into a one-dimensional one
  - Three dense layers were used, the last of which has 2 outputs, i.e. the number of classes (apples and oranges)

<b>1. DNN MODEL RESULTS</b>
 <pre>
>>> runfile('C:\\Users\\Lucian\\Desktop\\Masterat CTI\\Deep Learning\\applesandoranges.py', wdir='C:\\Users\\Lucian\\Desktop\\Masterat CTI\\Deep Learning')
Found 2014 files belonging to 2 classes.
Using 1612 files for training.
Found 2014 files belonging to 2 classes.
Using 402 files for validation.
Found 514 files belonging to 2 classes.
['mere', 'portocale']
81
26
(20, 80, 80, 3)
(20,)
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 rescaling (Rescaling)       (None, 80, 80, 3)         0         
                                                                 
 flatten (Flatten)           (None, 19200)             0         
                                                                 
 dense (Dense)               (None, 128)               2457728   
                                                                 
 dense_1 (Dense)             (None, 64)                8256      
                                                                 
 dense_2 (Dense)             (None, 2)                 130       
                                                                 
=================================================================
Total params: 2,466,114
Trainable params: 2,466,114
Non-trainable params: 0
_________________________________________________________________
None
Epoch 1/10
81/81 [==============================] - 8s 57ms/step - loss: 1.2597 - accuracy: 0.7270 - val_loss: 0.3973 - val_accuracy: 0.8333
Epoch 2/10
81/81 [==============================] - 5s 58ms/step - loss: 0.4647 - accuracy: 0.8282 - val_loss: 1.1599 - val_accuracy: 0.6617
Epoch 3/10
81/81 [==============================] - 7s 82ms/step - loss: 0.4678 - accuracy: 0.8213 - val_loss: 0.3498 - val_accuracy: 0.8408
Epoch 4/10
81/81 [==============================] - 5s 65ms/step - loss: 0.6720 - accuracy: 0.7816 - val_loss: 0.3673 - val_accuracy: 0.8383
Epoch 5/10
81/81 [==============================] - 5s 59ms/step - loss: 0.3871 - accuracy: 0.8418 - val_loss: 0.3537 - val_accuracy: 0.8557
Epoch 6/10
81/81 [==============================] - 4s 46ms/step - loss: 0.2951 - accuracy: 0.8778 - val_loss: 0.3389 - val_accuracy: 0.8657
Epoch 7/10
81/81 [==============================] - 4s 48ms/step - loss: 0.3087 - accuracy: 0.8828 - val_loss: 0.5151 - val_accuracy: 0.7886
Epoch 8/10
81/81 [==============================] - 3s 40ms/step - loss: 0.2771 - accuracy: 0.8902 - val_loss: 0.3883 - val_accuracy: 0.8507
Epoch 9/10
81/81 [==============================] - 3s 36ms/step - loss: 0.2335 - accuracy: 0.9069 - val_loss: 0.3817 - val_accuracy: 0.8308
Epoch 10/10
81/81 [==============================] - 3s 36ms/step - loss: 0.2102 - accuracy: 0.9175 - val_loss: 0.4666 - val_accuracy: 0.8060
(514, 80, 80, 3)
(514,)
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 rescaling (Rescaling)       (None, 80, 80, 3)         0         
                                                                 
 flatten (Flatten)           (None, 19200)             0         
                                                                 
 dense (Dense)               (None, 128)               2457728   
                                                                 
 dense_1 (Dense)             (None, 64)                8256      
                                                                 
 dense_2 (Dense)             (None, 2)                 130       
                                                                 
=================================================================
Total params: 2,466,114
Trainable params: 2,466,114
Non-trainable params: 0
_________________________________________________________________
None
17/17 - 0s - loss: 0.4772 - accuracy: 0.8191 - 154ms/epoch - 9ms/step
0.8190661668777466

 </pre>

![image](https://github.com/TrifanLucian/DL-Image-Classification/assets/111199896/f5bf9af4-d66d-4494-ab41-ce5ced4ed7ef)

Confusion matrix:
![image](https://github.com/TrifanLucian/DL-Image-Classification/assets/111199896/c6fdce2d-e430-4065-ab3d-376abe76fb02)

<pre>
  Cod test:
path = r'C:\Users\Lucian\Desktop\Masterat CTI\Deep Learning\dataset_applesandoranges\valid\mere\n07740461_41.jpg'
img = tf.keras.utils.load_img(path, target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions)
print(predictions)
print(score)
  
test result:
1/1 [==============================] - 0s 43ms/step
[[9.9998021e-01 1.9794745e-05]]
tf.Tensor([[0.73105085 0.2689492 ]], shape=(1, 2), dtype=float32)

</pre>

<b>2. CNN MODEL RESULTS</b>
<pre>
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['C:\\Users\\Lucian\\Desktop\\Masterat CTI\\Deep Learning', 'C:\\Users\\Lucian\\PycharmProjects'])
PyDev console: starting.
Python 3.10.6 (tags/v3.10.6:9c7b4bd, Aug  1 2022, 21:53:49) [MSC v.1932 64 bit (AMD64)] on win32
runfile('C:\\Users\\Lucian\\Desktop\\Masterat CTI\\Deep Learning\\applesandoranges.py', wdir='C:\\Users\\Lucian\\Desktop\\Masterat CTI\\Deep Learning')
Found 2014 files belonging to 2 classes.
Using 1612 files for training.
Found 2014 files belonging to 2 classes.
Using 402 files for validation.
Found 514 files belonging to 2 classes.
['mere', 'portocale']
81
26
(20, 80, 80, 3)
(20,)
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 rescaling (Rescaling)       (None, 80, 80, 3)         0         
                                                                 
 conv2d (Conv2D)             (None, 80, 80, 32)        896       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 40, 40, 32)       0         
 )                                                               
                                                                 
 flatten (Flatten)           (None, 51200)             0         
                                                                 
 dense (Dense)               (None, 128)               6553728   
                                                                 
 dense_1 (Dense)             (None, 2)                 258       
                                                                 
=================================================================
Total params: 6,554,882
Trainable params: 6,554,882
Non-trainable params: 0
_________________________________________________________________
None
Epoch 1/10
81/81 [==============================] - 19s 202ms/step - loss: 0.5152 - accuracy: 0.8362 - val_loss: 0.2392 - val_accuracy: 0.9030
Epoch 2/10
81/81 [==============================] - 13s 162ms/step - loss: 0.1950 - accuracy: 0.9318 - val_loss: 0.2942 - val_accuracy: 0.8905
Epoch 3/10
81/81 [==============================] - 13s 167ms/step - loss: 0.1164 - accuracy: 0.9615 - val_loss: 0.3024 - val_accuracy: 0.8930
Epoch 4/10
81/81 [==============================] - 15s 183ms/step - loss: 0.0863 - accuracy: 0.9727 - val_loss: 0.2270 - val_accuracy: 0.9005
Epoch 5/10
81/81 [==============================] - 16s 197ms/step - loss: 0.0611 - accuracy: 0.9820 - val_loss: 0.3268 - val_accuracy: 0.8980
Epoch 6/10
81/81 [==============================] - 16s 193ms/step - loss: 0.0255 - accuracy: 0.9938 - val_loss: 0.2806 - val_accuracy: 0.9030
Epoch 7/10
81/81 [==============================] - 14s 177ms/step - loss: 0.0164 - accuracy: 0.9975 - val_loss: 0.3590 - val_accuracy: 0.8980
Epoch 8/10
81/81 [==============================] - 15s 181ms/step - loss: 0.0240 - accuracy: 0.9944 - val_loss: 0.2937 - val_accuracy: 0.9080
Epoch 9/10
81/81 [==============================] - 15s 189ms/step - loss: 0.0204 - accuracy: 0.9957 - val_loss: 0.3107 - val_accuracy: 0.9104
Epoch 10/10
81/81 [==============================] - 18s 225ms/step - loss: 0.0051 - accuracy: 0.9988 - val_loss: 0.4811 - val_accuracy: 0.8955
(514, 80, 80, 3)
(514,)
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 rescaling (Rescaling)       (None, 80, 80, 3)         0         
                                                                 
 conv2d (Conv2D)             (None, 80, 80, 32)        896       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 40, 40, 32)       0         
 )                                                               
                                                                 
 flatten (Flatten)           (None, 51200)             0         
                                                                 
 dense (Dense)               (None, 128)               6553728   
                                                                 
 dense_1 (Dense)             (None, 2)                 258       
                                                                 
=================================================================
Total params: 6,554,882
Trainable params: 6,554,882
Non-trainable params: 0
_________________________________________________________________
None
17/17 - 1s - loss: 0.4182 - accuracy: 0.9047 - 935ms/epoch - 55ms/step
0.9046692848205566
</pre>
![image](https://github.com/TrifanLucian/DL-Image-Classification/assets/111199896/1683fb5e-b2f9-488a-ac08-00a6f31322c8)

Confusion matrix:
![image](https://github.com/TrifanLucian/DL-Image-Classification/assets/111199896/7d6c6315-fd9f-4498-a43c-e3b2e0278ace)

<pre>
  test code:
path = r'C:\Users\Lucian\Desktop\Masterat CTI\Deep Learning\dataset_applesandoranges\valid\mere\n07740461_41.jpg'
img = tf.keras.utils.load_img(path, target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions)
print(predictions)
print(score)

test result:
1/1 [==============================] - 0s 185ms/step
[[1.0000000e+00 2.5329834e-22]]
tf.Tensor([[0.7310586  0.26894143]], shape=(1, 2), dtype=float32)
</pre>

<b>Conclusion

Regarding the models used, we notice that the CNN architecture has a better accuracy compared to the DNN architecture. For the convolutional network the accuracy is 90.4%, and for the neural network the accuracy is 81.9%. This is due to the fact that the CNN model works better on images.</b>


