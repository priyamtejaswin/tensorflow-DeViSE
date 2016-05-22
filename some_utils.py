import pickle
import numpy as np
import tensorflow as tf

def find_total_vocab_size(caption_set):
    ####if a new seller joins, what does he have to do - shipping time, prices, shipping dates???
    #####pareto -drop shipping time, increase price - sensitivity will vary buy category
    print "total image in dataset:", len(caption_set.keys())
    check_stop  = lambda string: string.split() if string.strip()[-1]!='.' else string[:-1].split()

    caption_coll = []
    for cap_list in caption_set.itervalues():
        caption_coll.extend(cap_list)
    
    word_coll = []
    for cap in caption_coll:
        word_coll.extend(check_stop(cap))

    word_set = set(word_coll)

    print "total words:", len(word_coll)
    print "unique words:", len(word_set)


norm = lambda a: np.sqrt(np.sum(np.square(a)))
cosine = lambda a,b: np.sum(a * b)/(norm(a) * norm(b))
alpha = 0.1
loss_per_image_label = lambda image, label, app: max(0, alpha - cosine(image, label)) if app else max(0, alpha + cosine(image, label)) 

norm = lambda x: tf.sqrt(tf.reduce_sum(tf.square(x)))
cosine = lambda x, y: tf.reduce_sum(x * y)/(norm(x) * norm(y))
loss_per_image_label = lambda im, la, app: max(0, alpha - cosine(im, la)) if app else max(0, alpha + cosine(im, la))

if __name__ == '__main__':
    # pkl = pickle.load(open('pascal/image_to_captions.pkl'))
    # find_total_vocab_size(pkl)
    print loss_per_image_label(np.random.rand(1, 5), np.random.rand(1, 5), False)