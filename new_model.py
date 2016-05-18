## Once again, our quest for understanding the mysteries of deep learning continues,
## this time, exploring "deep" image search systems.
import os
import cv2
import skimage
from skimage import io
from collections import defaultdict
import pickle
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.models.rnn import rnn, rnn_cell

def make_tensor_placeholder(lol):
    # [break_point, sample_index, 0]
    r_val = np.array([ [[len(lol[i]) - 1, i, 0]] for i in range(len(lol)) ] ) 
    # print "r_val", r_val.shape
    return r_val

def crop_image(x, target_height=227, target_width=227, as_float=True):
    #image = skimage.img_as_float(skimage.io.imread(x)).astype(np.float32)
    image = io.imread(x)
    if as_float:
        image = skimage.img_as_float(image).astype(np.float32)

    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height,target_width))

    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (target_height, target_width))

def read_image(path):
     img = crop_image(path, target_height=224, target_width=224)
     if img.shape[2] == 4:
         img = img[: , : , : 3]

     img = img[None, ...]
     return img

def apply_to_zeros(lst, dtype=np.int64):
    inner_max_len = n_lstm_steps
    result = np.zeros([len(lst), inner_max_len], dtype)
    for i, row in enumerate(lst):
        for j, val in enumerate(row):
            result[i][j] = val
    return result

def process_captions(path_to_file, path_to_save=False):
    # http://www.flickr.com/photos/luckyrva/3229898555/
    # http://www.flickr.com/photos/salsaboy/3423509305/
    word_list = []
    split_caption = lambda line: line[:-1].strip().lower().split() if line[-1]=='.' else line.strip().lower().split()

    # First pass to create the vocabulary.
    with open(path_to_file) as fp:
        reader = csv.reader(fp, delimiter='\t')
        for row in reader:
            caption = row[-1].strip()
            tokens = split_caption(caption)
            word_list.extend(tokens)
    
    word_to_id = {v:i for i,v in enumerate(set(word_list), 1)}
    id_to_word = {v:k for k,v in word_to_id.iteritems()}
    word_to_id['<NUL>'] = 0
    id_to_word[0] = '<NUL>' 
    
    pickle.dump(word_to_id, open('word_to_id.pkl', 'w'))
    pickle.dump(id_to_word, open('id_to_word.pkl', 'w'))

    caption_features = defaultdict(list) 
    # Second pass to encode the words as tokens.
    with open(path_to_file) as fp:
        reader = csv.reader(fp, delimiter='\t')
        for row in reader:
            img_id = row[0]
            tokens = split_caption(row[-1].strip())
            caption_features[img_id].append([word_to_id[t] for t in tokens])
            
    return len(word_to_id), caption_features

def process_images(path_to_images, path_to_save=False):
    # http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html
    # http://vision.cs.uiuc.edu/pascal-sentences/
    vgg_path='/home/priyam/deep-learning/vgg16.tfmodel'

    with open(vgg_path) as f:
        fileContent = f.read()
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fileContent)

    images = tf.placeholder("float32", [1, 224, 224, 3], name="vvg_input_image")
    tf.import_graph_def(graph_def, input_map={"images":images})

    sess = tf.InteractiveSession()
    graph = tf.get_default_graph()
    image_features = {}

    for fname in os.listdir(os.path.join(os.getcwd(), path_to_images)):
        if '.jpg' in fname:
            path = os.path.join(os.getcwd(), path_to_images, fname)
            print path
            image_val = read_image(path)
            image_features[fname.split('.')[0]] = sess.run(graph.get_tensor_by_name("import/fc7_relu:0"), 
                feed_dict={images:image_val})

    sess.close()
    return image_features

def rnn_model(X, init_state, lstm_size, slicing_tensors):
    # X, input shape: (batch_size, input_vec_size, time_step_size)
    # print "X shape", X.get_shape().as_list()
    XT = tf.transpose(X, [1, 0, 2])  # permute time_step_size and batch_size
    # XT shape: (input_vec_size, batch_szie, time_step_size)
    # print "XT shape", XT.get_shape().as_list()

    XR = tf.reshape(XT, [-1, lstm_size]) # each row has input for each lstm cell (lstm_size)
    # XR shape: (input vec_size, batch_size)
    # print sess.run(num_steps)
    # print "XR shape", XR.get_shape().as_list()

    X_split = tf.split(0, n_lstm_steps, XR) # split them to time_step_size (28 arrays)
    # Each array shape: (batch_size, input_vec_size)
    # print "X_split"
    # print len(X_split)
    # print X_split

    # Make lstm with lstm_size (each input vector size)
    lstm = rnn_cell.BasicLSTMCell(lstm_size, forget_bias=1.0)

    # Get lstm cell output, time_step_size (28) arrays with lstm_size output: (batch_size, lstm_size)
    outputs, _states = rnn.rnn(lstm, X_split, initial_state=init_state)
    # print  "outputs", outputs[0].get_shape()
    outputs = tf.reshape(tf.concat(0, outputs), [n_lstm_steps, batch_size, dim_hidden])
    # Linear activation is NOT REQUIRED!!
    # Get the last output.
    # print "outputs"
    # print len(outputs)
    # print outputs

    # Slicing the appropriate output vectors from the <outputs>
    # sliced_outputs = [tf.slice(outputs[break_points[i]-1], slicing_lengths[i][0], slicing_lengths[i][1]) for i in range(batch_size)]
    slicing_tensors = [tf.squeeze(tsr) for tsr in tf.split(0, batch_size, slicing_tensors)]
    # print  "slicing_tensors", slicing_tensors[0].get_shape()
    sliced_outputs = [tf.slice(outputs, begin=tensor, size=[1, 1, 256]) for tensor in slicing_tensors]
    # for begin,size in slicing_lengths:
        # print tf.slice(outputs, begin, size)

    # return outputs[-1], lstm.state_size # State size to initialize the state
    return tf.squeeze(tf.concat(0, sliced_outputs)), lstm.state_size

######################################################################
################################training + testing code########################
######################################################################
dim_image = 4096
dim_embed = 256
dim_hidden = 256
n_words, caption_features = process_captions('search_data/search_captions.txt')
batch_size = len(caption_features['1'])
slicing_lengths = [ [(i, 0), (1, dim_embed)] for i in range(len(caption_features['1']))  ]
print "slicing_lengths", slicing_lengths
break_points = [len(x) for x in caption_features['1']]
print "break_points", break_points
n_lstm_steps = max(len(x) for x in caption_features['1'])
image_features = process_images('search_data')
print "n_lstm_steps", n_lstm_steps

slicing_tensors = tf.placeholder(tf.int32, [batch_size, 1, 3]) 

# Word embedding matrix
with tf.device("/cpu:0"):
    Wemb = tf.Variable(tf.random_uniform([n_words, dim_embed], -0.1, 0.1), name='Wemb')
bemb = tf.Variable(tf.zeros([dim_embed]), name='bemb')

# Image encoder and input
encode_img_W = tf.Variable(tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_img_W')
encode_img_b = tf.Variable(tf.zeros([dim_hidden]), name='encode_img_b')

image_pl = tf.placeholder(tf.float32, [1, dim_image], name="image_pl")
image_map = tf.matmul(image_pl, encode_img_W) + encode_img_b

# Word input
sentence_pl = tf.placeholder(tf.int32, [batch_size, n_lstm_steps], name="sentence_pl")
sentence_map = tf.nn.embedding_lookup(Wemb, sentence_pl) + bemb
print "sentence_map shape", sentence_map.get_shape()

# For the output of the lstm model - ONLY USE FOR LANGUAGE MODELLING!!
embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='embed_word_W')
embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')
 
# LSTM model??!!
sentence_output_pl = tf.placeholder(tf.float32, [None, dim_hidden], name="sentence_output_pl")
init_state = tf.placeholder(tf.float32, [None, 2*dim_hidden], name="init_state")
# steps = tf.placeholder(tf.int32, shape=[1], name="lstm_steps") # Sequence length passed during runtime - there must be a faster way...

py_x, state_size = rnn_model(sentence_map, init_state, dim_hidden, slicing_tensors)
l2_norm = lambda mat: tf.sqrt(tf.reduce_sum(tf.square(mat), 1))
cost = tf.reduce_mean(l2_norm(py_x - image_map))
train_op = tf.train.AdamOptimizer(0.001).minimize(cost)

######################################################################
################################Tensorflow graph session#######################
######################################################################

sess = tf.InteractiveSession()
# with tf.Session() as sess:
# try_transpose = lambda X: tf.reshape(tf.transpose(X, [1, 0, 2]), [-1, 256]) 
# dummy_x = try_transpose(sentence_map)
sess.run(tf.initialize_all_variables())
# print "image_map", sess.run(image_map, feed_dict={image_pl: image_features['1']}).shape
# print "sentence_map", sess.run(sentence_map, feed_dict={sentence_pl: np.matrix(caption_features['1'][0])}).shape
for i in range(50):
    _cost_ = 0
    for key in caption_features.keys():
        sess.run(train_op,
                    feed_dict={
                        init_state: np.zeros((batch_size, state_size)),
                        sentence_pl: np.array(apply_to_zeros(caption_features[key])),
                        image_pl: image_features[key],
                        slicing_tensors: make_tensor_placeholder(caption_features[key])
                    }   
            )
        _cost_ += sess.run(cost,
                    feed_dict={
                        init_state: np.zeros((batch_size, state_size)),
                        sentence_pl: np.array(apply_to_zeros(caption_features[key])),
                        image_pl: image_features[key],
                        slicing_tensors: make_tensor_placeholder(caption_features[key])
                    }   
            )
    print i, _cost_

    # print sess.run(py_x, 
    #                         feed_dict={
    #                             init_state: np.zeros((batch_size, state_size)),
    #                             sentence_pl: np.array(apply_to_zeros(caption_features['1']))

    #         }).shape

# Works till here!


sess.close()

# if __name__ == '__main__':
    # pass
    # process_captions('search_data/search_captions.txt')
    # process_images('search_data')