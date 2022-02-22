# Tensorflow is Deep Learning Package
import tensorflow as tf
# Keras is based on Tensorflow & It is High end API
import tensorflow.keras as keras

# scalar with tensor rank = 0 , shape = 0, dtype = int32 bytes.
tf.constant(10)

# Vector with rank = 1
const_2 = tf.constant([2., 4., 6., 8., 11., 15], dtype = 'float16')
print(const_2)

# scalar with tensor rank = 0, shape = 0, -128 to +127, default dtype = int8
const_1 = tf.constant(-128, dtype = 'int8')
print(const_1)

const_1.numpy()

const_1.shape

const_3 = tf.constant([[10, 20, 30],
                       [12, 19, 22],
                       [55, 66, 77]], dtype = 'int8')

print(const_3)

const_3.numpy()

const_3.shape

const_4 = tf.constant([[[10, 20, 15],
                       [12, 19, 33],
                       [55, 56, 77]],
                       [[10, 30, 32],
                        [12, 91, 88],
                        [127, 129, 180]]], dtype = 'int16')
print(const_4)

var_1 = tf.Variable(0, dtype = 'int8')
print(var_1)

# Vector Rank = 1
var_2 = tf.Variable([2., 4., 6., 17., 11., 15.], dtype = 'float16')
print(var_2)

var_3 = tf.Variable([[[10, 20, 15],
                      [12, 19, 33],
                      [55, 56, 77]],
                     
                     [[10, 30, 32],
                      [12, 91, 88],
                      [127, 129, 180]]], dtype = 'int8')
print(var_3)

var_3 = tf.Variable([[[10, 20, 15],
                      [12, 19, 33],
                      [55, 56, 77]],
                     [[13, 30, 32],
                      [11, 48, 38],
                      [27, 29, 16]],
                     [[10, 30, 32],
                      [12, 91, 88],
                      [127, 129, 180]]], dtype = 'int16')
print(var_3)

# Y = WT * X + bias
Weights = tf.Variable([[10],
                       [12],
                       [15],
                       [16],
                       [19],
                       [11]], dtype = 'float16')
print(Weights)

X = tf.Variable([[20],
                 [22],
                 [25],
                 [16],
                 [17],
                 [18]], dtype = 'float16')
print(X)

# Y = W * X
tf.tensordot(X, Weights, axes = 1)

# Y = WT * X
tf.tensordot(WT, X, axes = 1)

# Y = WT * X
tf.tensordot(WT, X, axes = 1)

--------------------------------------------------------------------------------
# Dense Neural Network (Feed-Forward Neural Network)
import tensorflow.keras as keras
keras.__version__

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
# Optimizers for Regression Model
from tensorflow.keras.optimizers import RMSprop

def build_model():
    # Sequential Neural Network - FeedForward NN
    model = Sequential()
    # Units = Num of Neurons (2 * pow(n)), input shape = num of features.
    model.add(Dense(units = 2, activation = 'relu', input_shape = [2]))
    # Hidden Layer - I
    model.add(Dense(units = 4, activation = 'relu'))
    # Output Layer 
    model.add(Dense(units = 1))
    # Model Compiler
    model.compile(loss = 'mean_squared_error', optimizer = 'RMSProp', metrics = ['mean_squared_error',
                                                                                 'mean_absolute_error'] )
    return model

model = build_model()

model.summary()

