# tensorflow-DeViSE
Attempts to understand deep learning and the Tensorflow RNN api by implementing a (very)crude version of the [Google DeViSE paper(2013)](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41473.pdf).<br><br>
Series of events which led to this repo:<br>
1. Tried to understand theano.<br>
2. Tried to hack using theano.<br>
3. Tried to hack using tensorflow.

The paper's objective (as described in Section 3) is to *"leverage semantic knowledge learned in the text domain, and transfer it to a model trained for visual object recognition"*, which is exactly what I did...sort of.

I was developing/testing this on my Dell laptop which has a ATI graphics card (so no CUDA). My definition of success for this project was to make something which (1)works and (2)decreases the loss. I'm putting the code online because I could not find beginner level code for using/hacking recurrent neural networks. This "work" is a result of repeatedly editing(and breaking) the RNN language model examples available with tensorflow. Hopefully, someone can use this to start their own RNN experiments.<br>

### How this works:
The idea was to make a image search engine: given a query(sequence of tokens), return the candidate set of images sorted from most appropriate to least appropriate. Simple enough(**sarcasm**). All I had to do was to encode the entire query into a single vector of fixed length, encode the images into a single vector of same length and ensure that the encoding process can capture the ground truth. By capturing the ground truth, I mean that appropriate <image, query> pairs are closer in the embedded dimension space and inappropriate <image, query> pairs are far apart. Concretely, if there was an image of a dog drinking water, its "appropriate" query pair could be "dog is driking water" and an "inappropriate" query pair could be "man driving a car". The encoding process must ensure that the appropriate <image, query> vector pair is closer than the inappropriate <image, query> pair.<br>

### Encoding query strings:
This is a two step process. First, I convert the word into vectors(*see what I did there*). The size of these vectors(or word-embeddings) is not related to the size of the encoded images. The paper suggests to train a language model which learns these word embeddings from scratch. Instead I use [pre-trained Stanford Glove word embeddings](http://nlp.stanford.edu/projects/glove/). I chose this over [Google's Word2Vec](https://code.google.com/archive/p/word2vec/) because it offered a better coverage of my training vocabulary. More on this [here](https://groups.google.com/forum/#!msg/word2vec-toolkit/lxbl_MB29Ic/kvsdSeDXsYIJ).

Once the words have been encoded as vectors, I had to "condense" the list of vectors into a single vector. For this I use a Recurrent Neural Network with an LSTM cell. [Colah's blog](http://colah.github.io/) is a great resource for understanding related topics. I used [this](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) to learn more about RNNs. [This reddit post](https://www.reddit.com/r/MachineLearning/comments/3sok8k/tensorflow_basic_rnn_example_with_variable_length/) got me started with basic RNN code using the tensorflow RNN api. I consider the "intent/meaning" of a query to be the output returned by the RNN. The dimensions of this output should be equal to the dimensions of encoded images.<br>

### Encoding images:
Once again, a two step process. First, I extract the "best possible" feature vector from the image. The paper suggests to train a visual object recognition system based on the ILSVRC 2012 winner. Instead, I use [VGG16](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) pre-trained weights for tensorflow, which can be downloaded from [here](https://drive.google.com/file/d/0B5o40yxdA9PqSGtVODN0UUlaWTg/view). The features are extracted from the last fully connected layer of the network.

Once the image feautre vector has been extracted, a linear transformation maps the vector to new dimension space. The size of the encoded images should be the same as the size of the encoded queries. 

### Loss function:
Once encoded, the query string and the image have equal dimensions. The loss function proposed in the paper is a hinge rank loss described in Section 3.3. Its similar to hinge loss with the addition of contrastive(or inappropriate) pairs: at every training epoch, the weights are updated based on how close appropriate training pairs are and how far inappropriate pairs are.

The purpose of the model is to minimize this loss function and update the weights of all parameters using gradient descent. I used the [Pascal dataset](http://vision.cs.uiuc.edu/pascal-sentences/) for development. It contains 5 captions per image for 20 different catagories of objects; each catagory had 50 images. While developing this, I used a subset(150 images) of dogs, cats and birds.

### Code:
All the code is in `new_model.py`. The hyperparameters are global variables; a main function creates the model and starts training. The code for the model and tranining is half of all the code in the file, the rest is for processing data and handling I/O. I've tried to add comments regularly; it should be simple enough to read starting from the `main()` function. If you're feeling up to the task of hacking this for your own experiments(best of luck!!!), here's a list of python modules you'll need to get started:
- tensorflow 0.8
- numpy 1.11
- cv2 2.4.8
- skimage 0.9.3

I've seen boilerplate code where people have used [Caffe](http://caffe.berkeleyvision.org/tutorial/) for [extracting image features](https://github.com/BVLC/caffe/blob/master/examples/00-classification.ipynb) using AlexNet, CaffeNet etc. Please feel free to use that if its easier; that part of the pipeline is only used for feature extraction(it is not trained/updated during backprop so it should be easy to substitute). 

### To do:
- Add utility/interface for querying images once the model has been trained.
- Add a better way to evaluate model training(accuracy, recall etc).
