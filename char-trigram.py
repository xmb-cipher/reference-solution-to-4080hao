#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

import numpy, logging, argparse
import tensorflow as tf
logger = logging.getLogger()


def build_vocab( filename ):
	with open( filename, 'rb' ) as f:
		vocab = set([ w for s in f for w in s.split() ])
	vocab.add( '<s>' )
	vocab.add( '</s>' )
	word2idx = dict( zip( sorted(vocab), range(len(vocab)) ) )
	max_length = max( [ len(w) for w in word2idx ] )
	return word2idx, max_length
		   

################################################################################


def batch_consructor( filename, word2idx, batch_size = 512, shuffle = False ):
	idx2matrix = [ None ] * len(word2idx)
	for w in word2idx:
		idx2matrix[word2idx[w]] = [0] + [ord(c) for c in w] + [0] * (max_length - len(w))
	idx2matrix = numpy.asarray( idx2matrix, dtype = numpy.int32 )

	with open( filename, 'rb' ) as f:
		numeric = [ [ word2idx[w] for w in \
			['<s>'] * 2 + s.split() + ['</s>'] ] for s in f ]
	ngram = numpy.asarray([ s[i: i + 3] for s in numeric for i in xrange(len(s) - 2) ])

	if shuffle:
		numpy.random.shuffle( ngram )

	for i in xrange(0, ngram.shape[0], batch_size):
		yield ngram[i: i + batch_size, 0].reshape([-1]),\
			  ngram[i: i + batch_size, 0].reshape([-1]),\
			  idx2matrix[ngram[i: i + batch_size, 0].reshape([-1])], \
			  idx2matrix[ngram[i: i + batch_size, 1].reshape([-1])], \
			  ngram[i: i + batch_size, -1].reshape([-1])


################################################################################


if __name__ == '__main__':
	logging.basicConfig( format = '%(asctime)s : %(levelname)s : %(message)s', 
						 level= logging.INFO)

	parser = argparse.ArgumentParser()
	parser.add_argument( '--basename', type = str, default = 'ptb' )
	parser.add_argument( '--n_gram', type = int, default = 3 )
	parser.add_argument( '--n_batch_size', type = int, default = 256 )
	parser.add_argument( '--layer_size', type = str, default = '512,512' )
	parser.add_argument( '--n_embedding', type = int, default = 64 )
	parser.add_argument( '--learning_rate', type = float, default = 0.128 )
	parser.add_argument( '--momentum', type = float, default = 0.9 )
	parser.add_argument( '--kernel_height', type = str, default = '2,3,4,5,6,7,8,9' )
	parser.add_argument( '--kernel_depth', type = str, 
						 default = ','.join( ['16'] * 8 ) )  #'32,32,32,32,32,32,32,32' )
	args = parser.parse_args()
	
	logger.info( args )

	basename = args.basename
	n_gram = args.n_gram
	layer_size = args.layer_size
	n_embedding = args.n_embedding
	n_batch_size = args.n_batch_size
	learning_rate = args.learning_rate
	momentum = args.momentum
	kernel_height = [ int(x) for x in args.kernel_height.split(',') ]
	kernel_depth = [ int(x) for x in args.kernel_depth.split(',') ]
	assert len(kernel_height) == len(kernel_depth)

	word2idx, max_length = build_vocab( basename + '.train.txt' )
	logger.info( 'vocabulary built (max-length: %d)' % max_length )

	n_vocab = len(word2idx)
	n_in = [ 4 * sum(kernel_depth) ] + [ int(s) for s in layer_size.split(',') ]
	n_out = n_in[1:] + [ n_vocab ]
	assert len(n_in) == len(n_out)

	logger.info( 'input size:  %s' % str(n_in) )
	logger.info( 'output size: %s' % str(n_out) )

	################################################################################

	char_idx_1 = tf.placeholder( tf.int32, [None, None], name = 'char-idx-1' )
	char_idx_2 = tf.placeholder( tf.int32, [None, None], name = 'char-idx-2' )
	word_idx_1 = tf.placeholder( tf.int32, [None], name = 'word-idx-1' )
	word_idx_2 = tf.placeholder( tf.int32, [None], name = 'word-idx-2' )
	target = tf.placeholder( tf.int64, [None], name = 'target' )
	lr = tf.placeholder( tf.float32, [] )
	keep_prob = tf.placeholder( tf.float32, [] )

	projection = tf.Variable( tf.random_uniform( [128, n_embedding],
												  minval = -0.032, maxval = 0.032 ) )

	kernels = [ tf.Variable( tf.random_uniform( [h, n_embedding, 1, d], 
												minval = -0.032, maxval = 0.032 ) ) for \
				(h, d) in zip( kernel_height, kernel_depth ) ]

	kernel_bias = [ tf.Variable( tf.random_uniform( [d], 
													minval = -0.032, maxval = 0.032 ) ) for \
				    d in kernel_depth ]

	word_embedding = tf.Variable( tf.random_uniform( [n_vocab, sum(kernel_depth)],
								  minval = -0.032, maxval = 0.032 ) )

	W = [ tf.Variable( tf.random_uniform( [i, o] , 
										  minval = -0.032, maxval = 0.032 ) ) for \
				(i, o) in zip( n_in, n_out ) ]

	b = [ tf.Variable( tf.random_uniform( [o], minval = -0.032, maxval = 0.032 ) ) for o in n_out ]

	logger.info( 'placeholder & variable defined' )

	################################################################################

	char_cube_1 = tf.expand_dims( tf.gather( projection, char_idx_1 ), 3 )	
	char_conv_1 = [ tf.reduce_max( tf.nn.relu( tf.nn.conv2d( char_cube_1, kk, [1, 1, 1, 1], 'VALID' ) + bb ),
						   	   	   			   reduction_indices = [1, 2] ) \
					for kk,bb in zip(kernels, kernel_bias) ]

	char_cube_2 = tf.expand_dims( tf.gather( projection, char_idx_2 ), 3 )	
	char_conv_2 = [ tf.reduce_max( tf.nn.relu( tf.nn.conv2d( char_cube_2, kk, [1, 1, 1, 1], 'VALID' ) + bb ),
						   	   	   			   reduction_indices = [1, 2] ) \
					for kk,bb in zip(kernels, kernel_bias) ]

	word_project_1 = tf.gather( word_embedding, word_idx_1 )
	word_project_2 = tf.gather( word_embedding, word_idx_2 )

	layer_output = [ tf.nn.dropout( tf.concat( 1, \
						char_conv_1 + char_conv_2 + [word_project_1, word_project_2] ), keep_prob ) ]
	for i in xrange( len(W) ):
		layer_output.append( tf.matmul(layer_output[-1], W[i]) + b[i] )
		if i < len(W) - 1:
			layer_output[-1] = tf.nn.dropout( tf.nn.relu( layer_output[-1] ), keep_prob )
	xent = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( layer_output[-1], target ) )

	train_step = [ tf.train.MomentumOptimizer( lr, momentum, use_locking = True ).minimize( xent ) ]
	# train_step = [ tf.train.MomentumOptimizer( lr, momentum, use_locking = False )\
	# 			     .minimize( xent, var_list = W + b ),
	# 			   tf.train.MomentumOptimizer( lr / 2, momentum, use_locking = True )\
	# 			   	 .minimize( xent, var_list = kernels + [ projection ] ) ]

	logger.info( 'computational graph built' )

	################################################################################

	# sess = tf.Session( config = tf.ConfigProto( intra_op_parallelism_threads = 4 ) )
	config = tf.ConfigProto(  gpu_options = tf.GPUOptions( per_process_gpu_memory_fraction = 0.72 ) )
	sess = tf.Session( config = config )
	sess.run( tf.initialize_all_variables() )

	prev_ppl, decay_started, drop_rate = 6479745038, False, 0.256

	if not decay_started:
		saver = tf.train.Saver()

	for n_epoch in xrange(16):
		logger.info( 'epoch %2d, learning-rate: %f' % (n_epoch + 1, learning_rate) )
		saver.save( sess, 'cache' )

		cost, cnt = 0, 0
		for wi1, wi2, idx1, idx2, t in batch_consructor( basename + '.train.txt', 
											   			 word2idx, n_batch_size, True ):
			if t.shape[0] == n_batch_size:
				c = sess.run( train_step + [ xent ], 
							  feed_dict = { char_idx_1: idx1,
								  		    char_idx_2: idx2,
								  		    word_idx_1: wi1,
								  		    word_idx_2: wi2,
								  			target: t, 
								  			lr: learning_rate,
								  			keep_prob: 1 - drop_rate } )[-1]
			else:
				c = sess.run( xent, 
							  feed_dict = { char_idx_1: idx1,
								  			char_idx_2: idx2,
								  			word_idx_1: wi1,
								  		    word_idx_2: wi2,
								  			target: t,
								  			keep_prob: 1 } )
			cost += c * t.shape[0]
			cnt += t.shape[0]
			if cnt % n_batch_size == 100:
				logger.info( '%d examples processed' % cnt )
		train_ppl = numpy.e ** (cost / cnt)
		
		################################################################################

		cost, cnt = 0, 0
		for wi1, wi2, idx1, idx2, t in batch_consructor( basename + '.valid.txt', 
											   			 word2idx, n_batch_size, False ):
			c = sess.run( xent, 
						  feed_dict = { char_idx_1: idx1,
							  			char_idx_2: idx2,
							  			word_idx_1: wi1,
								  		word_idx_2: wi2,
							  			target: t,
							  			keep_prob: 1 } )
			cost += c * t.shape[0]
			cnt += t.shape[0]
		valid_ppl = numpy.e ** (cost / cnt)

		################################################################################

		cost, cnt = 0, 0
		for wi1, wi2, idx1, idx2, t in batch_consructor( basename + '.test.txt', 
											   			 word2idx, n_batch_size, False ):
			c = sess.run( xent, 
						  feed_dict = { char_idx_1: idx1,
							  			char_idx_2: idx2,
							  			word_idx_1: wi1,
							  			word_idx_2: wi2,
							  			target: t,
							  			keep_prob: 1 } )
			cost += c * t.shape[0]
			cnt += t.shape[0]
		test_ppl = numpy.e ** (cost / cnt)

		################################################################################

		logger.info( 'perplexity: %f (train), %f (valid), %f (test)', train_ppl, valid_ppl, test_ppl )

		if valid_ppl > prev_ppl or decay_started:
			learning_rate *= 0.5
			if not decay_started:
				saver.restore( sess, 'cache' )
				decay_started = True
		else:
			prev_ppl = valid_ppl
		drop_rate *= 0.5 ** (1./4)
