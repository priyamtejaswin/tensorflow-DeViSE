# tensorflow-DeViSE
Attempts to understand deep learning and the Tensorflow RNN api by implementing a (very)crude version of the [Google DeViSE paper(2013)](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41473.pdf).<br><br>
Series of events which led to this repo:<br>
1. Tried to understand theano.<br>
2. Tried to hack using theano.<br>
3. Tried to hack using tensorflow.<br><br>

The paper's objective (as described in Section 3) is to *"leverage semantic knowledge learned in the text domain, and transfer it to a model trained for visual object recognition"*, which is exactly what I did...sort of.<br><br>

Before you start ranting about my software development skills, I want to reiterate that its a crude implementation of the paper and also, I was developing/testing this on my Dell laptop which has a ATI graphics card (bye bye CUDA). My definition of success for this project was to make something which (1)works and (2)decreases the error. I'm putting the code online because I could not find beginner level code for using/hacking recurrent neural networks. This "work" is a result of repeatedly editing(and breaking) the RNN language model examples available with tensorflow. Hopefully, someone can use this to start their own RNN experiments.

### How this works:
The idea was to make a image search engine: given a query(sequence of tokens), return the candidate set of images sorted from most appropriate to least appropriate. Simple enough(**sarcasm**). All I had to do was to encode the entire query into a single vector of fixed length, encode the images into a single vector of same length and ensure that the encoding process can capture the ground truth. By capturing the ground truth, I mean that appropriate <image, query> pairs are closer in the embedded dimension space and inappropriate <image, query> pairs are far apart. Concretely, if there was an image of a dog drinking water, its "appropriate" query pair could be "dog is driking water" and an "inappropriate" query pair could be "man driving a car". The encoding process must ensure that the appropriate <image, query> vector pair is closer than the inappropriate <image, query> pair.<br><br>

### Encoding query strings:<br>
This is a two step process. First, I convert the word into vectors(*see what I did there*). The size of these vectors(or word-embeddings) is not related to the size of the encoded images. The paper suggests to train a language model which learns these word embeddings from scratch. Instead I use [pre-trained Stanford Glove word embeddings](http://nlp.stanford.edu/projects/glove/). I chose this over [Google's Word2Vec](https://code.google.com/archive/p/word2vec/) because it offered a better coverage of my training vocabulary. More on this [here](https://groups.google.com/forum/#!msg/word2vec-toolkit/lxbl_MB29Ic/kvsdSeDXsYIJ).<br><br>

Once the words have been encoded as vectors, I had to "condense" the list of vectors into a single vector. For this I use a Recurrent Neural Network with an LSTM cell. [Colah's blog](http://colah.github.io/) is a great resource for understanding related topics. I used [this](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) to learn more about RNNs. [This reddit post](https://www.reddit.com/r/MachineLearning/comments/3sok8k/tensorflow_basic_rnn_example_with_variable_length/) got me started with basic RNN code using the tensorflow RNN api. I consider the "intent/meaning" of a query to be the output returned by the RNN. The dimensionality of this output should be equal to the dimensionality of encoded images.

### Encoding images:<br>
Once again, a two step process. First, I extract the "best possible" feature vector from the image. The paper suggests to train a visual object recognition system based on the ILSVRC 2012 winner. Instead, I use [VGG16](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) pre-trained weights for tensorflow, which can be downloaded from [here](https://drive.google.com/file/d/0B5o40yxdA9PqSGtVODN0UUlaWTg/view). The features are extracted from the last fully connected layer of the network<br><br>

Once the image feautre vector has been extracted, a linear transformation maps the vector to new dimension space. The size of the encoded images should be the same as the size of the encoded queries. 

