#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

"""
Author      : Mingbin Xu (mingbin.xu@gmail.com)
Filename    : char2vec.py
Last Update : Jun 17, 2016
Description : N/A
Website     : https://wiki.eecs.yorku.ca/lab/MLL/

Copyright (c) 2016 iNCML (author: Mingbin Xu)
License: MIT License (see ../LICENSE)
"""

import numpy, logging, argparse, cPickle
import tensorflow as tf
from Char2VecUtil import *
logger = logging.getLogger()


if __name__ == '__main__':
    logging.basicConfig( format = '%(asctime)s : %(levelname)s : %(message)s', 
                         level= logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument( 'filename', type = str )
    parser.add_argument( '--mode', type = str, default = 'construct',
                         choices = [ 'train', 'construct', 'evaluate' ] )
    parser.add_argument( '--top_k', type = int, default = 50000 )
    parser.add_argument( '--n_batch_size', type = int, default = 64 )
    parser.add_argument( '--n_embedding', type = int, default = 64 )
    parser.add_argument( '--n_window', type = int, default = 4 )
    parser.add_argument( '--n_negative', type = int, default = 128 )
    parser.add_argument( '--learning_rate', type = float, default = 0.256 )
    parser.add_argument( '--kernel_height', type = str, default = '2,3,4,5,6,7,8,9' )
    parser.add_argument( '--kernel_depth', type = str, 
                         default = ','.join( ['16'] * 8 ) )  #'32,32,32,32,32,32,32,32' )
    args = parser.parse_args()

    filename = args.filename
    mode = args.mode

    if mode == 'train':
        with open( 'conv-config', 'wb' ) as fp:
            cPickle.dump( args, fp )
    else:
        with open( 'conv-config', 'rb' ) as fp:
            args = cPickle.load( fp )
        args.mode = mode
        args.filename = filename

    top_k = args.top_k
    n_batch_size = args.n_batch_size
    n_embedding = args.n_embedding
    n_window = args.n_window
    n_negative = args.n_negative
    learning_rate = args.learning_rate
    kernel_height = [ int(x) for x in args.kernel_height.split(',') ]
    kernel_depth = [ int(x) for x in args.kernel_depth.split(',') ]
    assert len(kernel_height) == len(kernel_depth)

    logger.info( args )

    ################################################################################

    if mode == 'train':
        data = batch_constructor( filename, top_k )
        logger.info( 'training data loaded' )

    ################################################################################

    char_idx = tf.placeholder( tf.int32, [None, None], name = 'char-idx' )
    target = tf.placeholder( tf.int64, [None, n_window * 2], name = 'target' )
    lr = tf.placeholder( tf.float32, [], name = 'learning-rate' )

    ################################################################################

    projection = tf.random_uniform( [128, n_embedding],
                                     minval = -0.032, maxval = 0.032 )
    projection = tf.Variable( projection, name = 'char-projection' )

    kernels = [ tf.Variable( tf.random_uniform( [h, n_embedding, 1, d], 
                                        minval = -0.032, maxval = 0.032 ) ) for \
                (h, d) in zip( kernel_height, kernel_depth ) ]

    kernel_bias = [ tf.Variable( tf.random_uniform( [d], 
                                        minval = -0.032, maxval = 0.032 ) ) for \
                    d in kernel_depth ]

    W = tf.Variable( tf.random_uniform( [top_k, sum(kernel_depth)] , 
                                         minval = -0.032, maxval = 0.032 ),
                     name = 'softmax-weight' )

    b = tf.Variable( tf.random_uniform( [top_k] , 
                                         minval = -0.032, maxval = 0.032 ),
                     name = 'softmax-bias' )

    ################################################################################

    char_cube = tf.expand_dims( tf.gather( projection, char_idx ), 3 )  
    char_conv = [ tf.reduce_max( tf.nn.relu( tf.nn.conv2d( char_cube, kk, [1, 1, 1, 1], 'VALID' ) + bb ),
                                             reduction_indices = [1, 2] ) \
                    for kk,bb in zip(kernels, kernel_bias) ]
    word_embedding = tf.concat( 1, char_conv )

    sample_idx, _, _ = ( tf.nn.fixed_unigram_candidate_sampler(
                            true_classes =  target, # tf.reshape( target, [-1, 1] ),
                            num_true = n_window * 2,
                            num_sampled = n_negative,
                            unique = True,
                            range_max = top_k,
                            distortion = 0.75,
                            unigrams = data.idx2freq if mode == 'train' else [1] * top_k ) )

    positive_w = tf.reshape( tf.gather( W, target ),
                             [-1, tf.shape(word_embedding)[1] ] )
    positive_b = tf.reshape( tf.gather( b, target ), [-1] )

    negative_w = tf.gather( W, sample_idx )
    negative_b = tf.reshape( tf.gather( b, sample_idx ), [-1] )

    repeat = tf.reshape( tf.tile( word_embedding, [1, 2 * n_window] ), 
                         [ -1, tf.shape(word_embedding)[1] ] )
    positive_logits = tf.reduce_sum( tf.mul( repeat, positive_w ), 
                                     1 ) + positive_b

    negative_logits = tf.matmul( word_embedding, 
                                 negative_w, 
                                 transpose_b = True ) + negative_b

    xent = tf.reduce_sum( tf.nn.sigmoid_cross_entropy_with_logits(
                positive_logits, tf.ones_like( positive_logits ) ) ) + \
           tf.reduce_sum( tf.nn.sigmoid_cross_entropy_with_logits(
                negative_logits, tf.zeros_like( negative_logits ) ) )
    xent = xent / tf.cast( tf.shape(target)[0], dtype = tf.float32 ) / (2 * n_window)

    optimizer = tf.train.GradientDescentOptimizer( lr )
    train_step = optimizer.minimize( xent, gate_gradients = optimizer.GATE_NONE )

    ################################################################################

    config = tf.ConfigProto(  gpu_options = tf.GPUOptions( per_process_gpu_memory_fraction = 0.72 ) )
    sess = tf.Session( config = config )
    sess.run( tf.initialize_all_variables() )
    saver = tf.train.Saver()

    if mode == 'train':
        cost, exam_freq = 0, 1000
        for step, (ix, t) in enumerate( 
                data.mini_batch_multi_thread( n_batch_size, n_window, True ) ):

            if step == 0:
                c = sess.run( xent, feed_dict = { char_idx: ix, target: t } )
                logger.info( 'error of one mini-batch before training: %f' % c )

            _, c = sess.run( [ train_step, xent ],
                              feed_dict = { char_idx: ix,
                                            target: t,
                                            lr: learning_rate } )
            cost += c
            # print step, c, ix.shape, t.shape
            if (step + 1) % exam_freq == 0:
                logger.info( 'step: %6d, learning-rate: %f, avg-cost: %f' % \
                    (step + 1, learning_rate, cost / exam_freq) )
                cost = 0
                learning_rate *= 0.5 ** (1./1024)
                saver.save( sess, 'conv-word2vec' )


    elif mode == 'construct':
        saver.restore( sess, 'conv-word2vec' )
        with open( filename, 'rb' ) as fp:
            words = [ x.strip() for x in fp.read().split() ]

        vectors = []
        for ix in word_batch( words, n_batch_size ):
            v = sess.run( word_embedding, feed_dict = { char_idx: ix } )
            vectors.append( v )
        vectors = numpy.concatenate( vectors, axis = 0 )

        logger.info( vectors.shape )

        with open( filename[:filename.rfind('.')] + '.word2vec', 'wb' ) as fp:
            numpy.int32(vectors.shape[0]).tofile( fp )
            numpy.int32(vectors.shape[1]).tofile( fp )
            vectors.tofile( fp )


    elif mode == 'evaluate':
        with open( filename + '.wordlist', 'rb' ) as fp:
            idx2word = numpy.asarray([ x.strip() for x in fp.read().split() ])

        with open( filename + '.word2vec', 'rb' ) as fp:
            shape = numpy.fromfile( fp, dtype = numpy.int32, count = 2 )
            vectors = numpy.fromfile( fp, dtype = numpy.float32 ).reshape( shape )
        logger.info( 'vectors loaded' )

        normalized_vectors = vectors / numpy.sqrt(numpy.square(vectors).sum(1)).reshape([-1,1])
        logger.info( 'vectors normalized' )

        similarity = numpy.dot( normalized_vectors, normalized_vectors.T )
        logger.info( 'similarity computed' )

        top = numpy.argsort( similarity, axis = 1, kind = 'heapsort' )
        logger.info( 'similarity sorted' )

        with open( filename + '.similarity', 'wb' ) as fp:
            for i, w in enumerate(idx2word):
                print >> fp, w.ljust(16, ' '), idx2word[top[i][-10:]].tolist()[::-1]

    sess.close()
