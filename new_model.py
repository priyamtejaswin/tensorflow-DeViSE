## Once again, our quest for understanding the mysteries of deep learning continues,
## this time, exploring "deep" image search systems.
import sys
import string
import time
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

def apply_to_zeros(lst, dtype=np.int32):
    inner_max_len = n_lstm_steps
    result = np.zeros([len(lst), inner_max_len], dtype)
    for i, row in enumerate(lst):
        for j, val in enumerate(row):
            result[i][j] = val
    return result

def split_caption(sentence):
    valid = string.letters + ' '
    s = ''
    for l in sentence.lower():
        if l in valid:
            s+=l
        else:
            s+=' '
    return s.strip().split()

def process_captions(path_to_w2id=False, list_of_captions=False, path_to_save=False):
    try:
        word_to_id = pickle.load(open(path_to_w2id))
    except:
        word_to_id = path_to_w2id
    # Second pass to encode the words as tokens.
    get_id = lambda w: word_to_id[w] if w in word_to_id else 0
    caption_features = []
    
    for caption in list_of_captions:
        tokens = split_caption(caption)
        caption_features.append([get_id(t) for t in tokens])
    return caption_features

def process_images(path_to_images, path_to_save=False):
    # http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html
    # http://vision.cs.uiuc.edu/pascal-sentences/
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
    sliced_outputs = [tf.slice(outputs, begin=tensor, size=[1, 1, dim_hidden]) for tensor in slicing_tensors]
    # for begin,size in slicing_lengths:
        # print tf.slice(outputs, begin, size)

    # return outputs[-1], lstm.state_size # State size to initialize the state
    # return tf.squeeze(tf.concat(0, sliced_outputs)), lstm.state_size
    return sliced_outputs, lstm.state_size

######################################################################
################################GLOBALS################################
######################################################################
vgg_path='/home/priyam/deep-learning/vgg16.tfmodel'
dim_image = 4096
dim_embed = dim_hidden = 300
caption_features = pickle.load(open('150_caption_features.pkl'))
vocab_size = len(pickle.load(open('word_to_id.pkl')))
batch_size = 20 + 1
num_con = 20
image_features = pickle.load(open('150_image_features.pkl'))
# slicing_lengths = [ [(i, 0), (1, dim_embed)] for i in range(len(caption_features['1']))  ]
# print "slicing_lengths", slicing_lengths
# break_points = [len(x) for x in caption_features['1']]
# print "break_points", break_points
n_lstm_steps = max(len(x) for x in caption_features)
margin = 0.1
max_epochs = 50
# image_features = process_images('search_data')
print "vocab_size", vocab_size
print "batch_size", batch_size
print "n_lstm_steps", n_lstm_steps
print "dim_embed", dim_embed
print "dim_image", dim_image
print "dim_hidden", dim_hidden
print "hinge loss margin:", margin
print "contrastive samples:", num_con
print "max_epochs:", max_epochs

def alt_cost(image, con_image, py_s):
    l2 = lambda x: tf.sqrt(tf.reduce_sum(tf.square(x)))
    l2norm = lambda x: x/l2(x)

    cap = tf.squeeze(py_s[0], [0])
    con_cap = tf.squeeze(tf.concat(0, py_s[1:]))

    cap = tf.tile(cap, (num_con, 1))
    image = tf.tile(image, (num_con, 1))

    image = l2norm(image)
    con_image = l2norm(con_image)
    cap = l2norm(cap)
    con_cap = l2norm(con_cap)

    cost_im = margin - tf.reduce_sum((image * cap), 1) + tf.reduce_sum((image * con_cap), 1)
    cost_im = cost_im * tf.maximum(cost_im, 0.0)
    cost_im = tf.reduce_sum(cost_im, 0)

    cost_s  = margin - tf.reduce_sum((cap * image), 1) + tf.reduce_sum((cap * con_image), 1)
    cost_s  = cost_s  * tf.maximum(cost_s, 0.0)
    cost_s  = tf.reduce_sum(cost_s,  0)

    cost = cost_im + cost_s
    return cost

######################################################################
##############################main-TRAINING FUNCTION########################
######################################################################

def main(pre_trained_WEs=False):
    slicing_tensors = tf.placeholder(tf.int32, [batch_size, 1, 3]) 
    # Word embedding matrix
    with tf.device("/cpu:0"):
        Wemb = tf.Variable(tf.random_uniform([vocab_size, dim_embed], -0.1, 0.1), name='Wemb')
    bemb = tf.Variable(tf.zeros([dim_embed]), name='bemb')

    # Image encoder and input
    encode_img_W = tf.Variable(tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_img_W')
    encode_img_b = tf.Variable(tf.zeros([dim_hidden]), name='encode_img_b')

    image_pl = tf.placeholder(tf.float32, [1, dim_image], name="image_pl")
    image_map = tf.matmul(image_pl, encode_img_W) + encode_img_b
    
    con_image_pl = tf.placeholder(tf.float32, [num_con, dim_image], "con_image_pl")
    con_image_map = tf.matmul(con_image_pl, encode_img_W) + encode_img_b

    # Word input
    sentence_pl = tf.placeholder(tf.int32, [batch_size, n_lstm_steps], name="sentence_pl")
    sentence_map = tf.nn.embedding_lookup(Wemb, sentence_pl) + bemb

    con_sentence_pl = tf.placeholder(tf.int32, [num_con, n_lstm_steps], name="con_sentence_pl")
    con_sentence_map = tf.nn.embedding_lookup(Wemb, con_sentence_pl) + bemb

    # For the output of the lstm model - ONLY USE FOR LANGUAGE MODELLING!!
    embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, vocab_size], -0.1, 0.1), name='embed_word_W')
    embed_word_b = tf.Variable(tf.zeros([vocab_size]), name='embed_word_b')
     
    # LSTM model??!!
    init_state = tf.placeholder(tf.float32, [None, 2*dim_hidden], name="init_state")
    # steps = tf.placeholder(tf.int32, shape=[1], name="lstm_steps") # Sequence length passed during runtime - there must be a faster way...

    py_x, state_size = rnn_model(sentence_map, init_state, dim_hidden, slicing_tensors)
    # l2_norm = lambda mat: tf.sqrt(tf.reduce_sum(tf.square(mat), 1))
    # cost = tf.reduce_mean(l2_norm(py_x - image_map))
    cost = alt_cost(image_map, con_image_map, py_x)
    train_op = tf.train.AdamOptimizer(0.001).minimize(cost)

    ######################################################################
    ################################Tensorflow graph session#######################
    ######################################################################
    select_random = lambda ls, num: np.random.choice(ls, num, replace=False)

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    if pre_trained_WEs:
        print "\nLoading pre-trained Word Embeddings from:", pre_trained_WEs
        ptr_wes = pickle.load(open(pre_trained_WEs))
        sess.run(Wemb.assign(ptr_wes))

    print "\n----All variables initialized...Commence model training----\n"
    
    for epoch in range(1, max_epochs+1):
        _cost_ = 0
        for i in range(len(image_features)):

            valid_caption_ids = caption_features[i*10 : (i+1)*10]
            cont_cap_candidates = [x for x in range(len(caption_features)) if x not in valid_caption_ids]
            cont_img_candidates = [x for x in range(len(image_features)) if x!=i]
           
            for j in range(len(valid_caption_ids)):
                cont_caption_ids = select_random(cont_cap_candidates, num_con)
                batch_captions = [valid_caption_ids[j]] + [caption_features[r] for r in cont_caption_ids]
                batch_caption_features = apply_to_zeros(batch_captions)
                batch_captions_breaks = make_tensor_placeholder(batch_captions)

                cont_image_ids = select_random(cont_img_candidates, num_con)
                cont_image_features = np.array([image_features[x] for x in cont_image_ids])

                # cfs = caption_features[i*5]
                # con_cfs = select_random([ x for x in range(len(caption_features)) if x not in range(i, (i+1)*5)], 5)
                # c = [cfs] + [caption_features[r] for r in con_cfs]
                # c_feats = apply_to_zeros(c)
                # breaks = make_tensor_placeholder(c)
                # # print c_feats
                # # print breaks
                # c_images_ix = select_random([x for x in range(len(image_features)) if x!=i], 5)
                # c_ims = np.array([image_features[x] for x in c_images_ix])

                sess.run(train_op,
                            feed_dict={
                                init_state: np.zeros((batch_size, state_size)),
                                sentence_pl: batch_caption_features,
                                image_pl: np.matrix(image_features[i]),
                                slicing_tensors: batch_captions_breaks,
                                con_image_pl: cont_image_features
                            }   
                    )

                _cost_ += sess.run(cost,
                            feed_dict={
                                init_state: np.zeros((batch_size, state_size)),
                                sentence_pl: batch_caption_features,
                                image_pl: np.matrix(image_features[i]),
                                slicing_tensors: batch_captions_breaks,
                                con_image_pl: cont_image_features
                            }   
                    )

        print epoch,  _cost_

    # Works till here!
    sess.close()

def extract_all_image_features(path_to_images, image_subset=False, batch_size=1):
    with open(vgg_path) as f:
        fileContent = f.read()
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fileContent)

    images = tf.placeholder(tf.float32, [batch_size, 224, 224, 3], name="vvg_input")
    tf.import_graph_def(graph_def, input_map={"images":images})

    iSess = tf.InteractiveSession()
    graph = tf.get_default_graph()

    if batch_size==1:
        pass
    else:
        batch_of_images = []
        image_ids = []
        for fname in os.listdir(path_to_images):
            if image_subset:
                if fname in image_subset:
                    is_in_subset = True
                else:
                    is_in_subset = False
            else:
                is_in_subset = True # If subset is not specified, then every image is to be considered(or every image is in the subset)

            if is_in_subset and ('.jpg' in fname):
                path = os.path.join(os.getcwd(), path_to_images, fname)
                print path
                image_ids.append(fname)
                batch_of_images.append(read_image(path))

        batch_of_images = np.vstack(batch_of_images)
        print "Batch of images:", len(batch_of_images)
        image_features = []
        for i in range(0, len(batch_of_images), batch_size)[:-1]:
            print "processing images", i, "to", i+batch_size
            image_features.append(iSess.run(graph.get_tensor_by_name("import/fc7_relu:0"),  feed_dict={images:batch_of_images[i: i+batch_size]}))
            
    iSess.close()
    return image_ids, np.vstack(image_features)

def build_feature_pkls():
    topics = pickle.load(open('pascal/topic_to_images.pkl'))
    captions = pickle.load(open('pascal/image_to_captions.pkl'))

    subset_images = []
    for key in topics.keys():
        if key in ['dog', 'cat', 'bird']:
            subset_images.extend(topics[key])

    list_of_captions = []
    for c in subset_images:
        list_of_captions.extend(captions[c] )

    w2id = pre_process_captions(list_of_captions=list_of_captions)
    caption_features = process_captions(w2id, list_of_captions=list_of_captions)
    
    with open('150_caption_features.pkl', 'w') as fp:
        pickle.dump(caption_features, fp)
        fp.flush()

    # return

    t0 = time.time()
    image_ids, image_features = extract_all_image_features('pascal/pascal-sentences_files', subset_images, 20)
    t1 = time.time()
    print "image features:", len(image_features), image_features.shape
    print (t1-t0)/60, "mins\n"

    print image_ids
    pickle.dump(image_ids, open('image_ids.pkl', 'w'))
    pickle.dump(image_features, open('150_image_features', 'w'))

def pre_process_captions(list_of_captions):
    word_list = []
    # First pass to create the vocabulary.
    for caption in list_of_captions:
        tokens = split_caption(caption.strip())
        word_list.extend(tokens)

    # word_list = [w for w in word_list if w not in ['jackethalter', '<NUL>', 'fencepost']]
    word_to_id = {v:i for i,v in enumerate(set(word_list), 1)}
    id_to_word = {v:k for k,v in word_to_id.iteritems()}
    word_to_id['<NUL>'] = 0
    id_to_word[0] = '<NUL>' 
    print "Unique words:", len(word_to_id)
    
    pickle.dump(word_to_id, open('word_to_id.pkl', 'w'))
    pickle.dump(id_to_word, open('id_to_word.pkl', 'w'))

    print max(word_to_id.itervalues()), max(id_to_word.iterkeys())
    return word_to_id

if __name__ == '__main__':
    # pass
    # build_feature_pkls()
    main(pre_trained_WEs='glove_NO_UKN.npy')
