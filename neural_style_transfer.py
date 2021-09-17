# Works on Python 3 or conda_tensorflow2_p36 kernel

# Import statements

import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
import pprint
%matplotlib inline

# Load parameters from VGG model (model has already been trained on the very large ImageNet database)

pp = pprint.PrettyPrinter(indent=4)
img_size = 400
vgg = tf.keras.applications.VGG19(include_top=False,
                                  input_shape=(img_size, img_size, 3),
                                  weights='pretrained-model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

vgg.trainable = False
pp.pprint(vgg)

# Declare function that computes the content cost 

def compute_content_cost(content_output, generated_output):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C) 
    a_G -- tensor of dimension (1, n_H, n_W, n_C)
    
    Returns: 
    J_content -- scalar
    """
    a_C = content_output[-1]
    a_G = generated_output[-1]
    
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_C_unrolled = tf.reshape(a_C, [m ,n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, [m ,n_H * n_W, n_C])

    J_content = (1/ (4*n_H*n_W*n_C))*tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))
    
    return J_content
  
  # Construct gram matrix (used to calculate the cost for the style image)

def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """  
    
    GA = tf.matmul(A, tf.transpose(A))

    return GA
  
  # Computes style cost for a single layer

def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value
    """

    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_S = tf.reshape(tf.transpose(a_S), [n_C, n_H*n_W])
    a_G = tf.reshape(tf.transpose(a_G), [n_C, n_H*n_W])

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style_layer = (1/ (4*n_C*n_C* (n_H*n_W)*(n_H*n_W)))*tf.reduce_sum(tf.square(tf.subtract(GS,GG)))
    
    return J_style_layer
  
  
  # Layer to be used as the content layer

vgg.get_layer('block5_conv4').output

# Layers to be used to represent the style

STYLE_LAYERS = [
    ('block1_conv1', 0.2),
    ('block2_conv1', 0.2),
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.2)]

# Computes the overall style cost from several chosen layers

def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
    """
    Arguments:
    style_image_output -- the tensorflow model
    generated_image_output --
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers chosen to extract style from
                        - a coefficient for each of them
    
    Returns: 
    J_style -- tensor representing a scalar value
    """
    
    J_style = 0

    a_S = style_image_output[:-1]

    a_G = generated_image_output[:-1]
    for i, weight in zip(range(len(a_S)), STYLE_LAYERS):  
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])
        J_style += weight[1] * J_style_layer

    return J_style
  
  # Computes the total cost function

@tf.function()
def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost
    """
    J = alpha * J_content + beta * J_style

    return J
  
  # Load content image

content_image = np.array(Image.open("images/boulder.jpg").resize((img_size, img_size)))
content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))

print(content_image.shape)
imshow(content_image[0])
plt.show()

# Load style image

style_image =  np.array(Image.open("images/backbone2.jpg").resize((img_size, img_size)))
style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))

print(style_image.shape)
imshow(style_image[0])
plt.show()

# Randomly generate initial image (semi-randomly, it is skewed to match content image)

generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
noise = tf.random.uniform(tf.shape(generated_image), 0, 0.5)
generated_image = tf.add(generated_image, noise)
generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)

print(generated_image.shape)
imshow(generated_image.numpy()[0])
plt.show()

# Creates a vgg model that returns a list of intermediate output values

def get_layer_outputs(vgg, layer_names):
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model
  
  content_layer = [('block5_conv4', 1)]

vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)
print(vgg_model_outputs)

# Computes content cost

preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
a_C = vgg_model_outputs(preprocessed_content)
a_G = vgg_model_outputs(generated_image)
J_content = compute_content_cost(a_C, a_G)

print(J_content)

# Computes style cost

preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
a_S = vgg_model_outputs(preprocessed_style)
J_style = compute_style_cost(a_S, a_G)
print(J_style)

# Truncate all the pixels in the tensor to be between 0 and 1

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

# Converts the given tensor into a PIL image

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)
  
  # Train step

optimizer = tf.keras.optimizers.Adam(learning_rate=0.03)

@tf.function()
def train_step(generated_image):
    with tf.GradientTape() as tape:
        
        a_G = vgg_model_outputs(generated_image)
        
        J_style = compute_style_cost(a_S,a_G)

        J_content = compute_content_cost(a_C,a_G)
        
        J = total_cost( J_content, J_style, alpha = 10, beta =40 ) 

        
    grad = tape.gradient(J, generated_image)

    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(clip_0_1(generated_image))
    
    return J
  
  generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))

J1 = train_step(generated_image)
print(J1)


J2 = train_step(generated_image)
print(J2)

# Shows the generated image at some epochs

epochs = 1000
for i in range(epochs):
    train_step(generated_image)
    if i % 2 == 0:
        print(f"Epoch {i} ")
    if i % 100 == 0:
        image = tensor_to_image(generated_image)
        imshow(image)
        image.save(f"output/image_{i}.jpg")
        plt.show() 
