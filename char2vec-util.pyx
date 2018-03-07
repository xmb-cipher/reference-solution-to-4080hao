"""
Author      : Mingbin Xu (mingbin.xu@gmail.com)
Filename    : char2vec-util.pyx
Last Update : Jun 17, 2016
Description : N/A
Website     : https://wiki.eecs.yorku.ca/lab/MLL/

Copyright (c) 2016 iNCML (author: Mingbin Xu)
License: MIT License (see ../LICENSE)
"""


import numpy, re, random, logging, time
from Queue import Queue
from threading import Thread

cimport cython, numpy
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map as ordered_map
from cython.operator cimport dereference, preincrement

cdef extern from "<regex>" namespace "std" nogil:
    cdef cppclass regex:
        regex( string& s ) except +
    bint regex_match( string& s, regex& r )

logger = logging.getLogger()

################################################################################


cdef class batch_constructor(object):

    cdef vector[vector[string]] original
    cdef vector[vector[int]] numeric
    cdef readonly dict word2idx
    cdef readonly list idx2word
    cdef readonly list idx2freq
    cdef regex* number
    
    
    def __cinit__( self, filename, top_k = 50000 ):
        self.number = new regex( r"^(?=[^A-Za-z]+$).*[0-9].*$" )

        cdef int i, j, unk
        cdef string s
        cdef vector[string] words
        cdef dict word2cnt = {}
        last_report = time.time()
        
        with open( filename ) as fp:
            for line in fp:
                line = line.strip().split()

                for w in line:
                    s = w.lower()
                    if regex_match( s, self.number[0] ):
                        s = '<numeric>'
                    if word2cnt.has_key(s):
                        word2cnt[s] += 1
                    else:
                        word2cnt[s] = 1

                words = line
                self.original.push_back( words )

                if time.time() - last_report > 60:
                    logger.info( '%d sentence(s), %d word type(s) read' % \
                                 (self.original.size(), len(word2cnt)) )
                    last_report = time.time()

        logger.info( '%d sentence(s), %d word type(s) read' % \
                                 (self.original.size(), len(word2cnt)) )

        freq = [ (word2cnt[w], w) for w in word2cnt ]
        freq.sort( key = lambda x: (-x[0], x[1]) )
        logger.info( 'top 10 words: %s' % str([ w for _, w in freq[:10]]) )

        self.word2idx = dict( [(x[1],i) for i,x in enumerate(freq[:top_k - 3])] )
        self.word2idx['<s>'] = len(self.word2idx)
        self.word2idx['</s>'] = len(self.word2idx)
        self.word2idx['<unk>'] = len(self.word2idx)
        unk = self.word2idx['<unk>']

        self.idx2word = [ x[1] for x in freq[:top_k - 3] ]
        self.idx2word.append( '<s>' )
        self.idx2word.append( '</s>' )
        self.idx2word.append( '<unk>' )

        # with open( 'wordlist', 'wb' ) as fp:
        #     for w in self.idx2word:
        #         print >> fp, w

        self.idx2freq = [ x[0] for x in freq[:top_k - 3] ]
        self.idx2freq.extend( [max( 1, self.idx2freq[-1] / 10 )] * 3 )
        # self.idx2freq.append( sum( [ x[0] for x in freq[top_k - 1:] ]) )

        last_report = time.time()
        self.numeric.resize( self.original.size() )
        for i in range( self.original.size() ):
            self.numeric[i].resize( self.original[i].size() )
            for j in range( self.original[i].size() ):
                s = self.original[i][j]
                if regex_match( s, self.number[0] ):
                    s = '<numeric>'
                self.numeric[i][j] = self.word2idx.get( s, unk )
            if time.time() - last_report > 60:
                logger.info( '%d sentence(s) numericized' % i )
                last_report = time.time()

        logger.info( '%d sentence(s) numericized' % i )


    def __dealloc__( self ):
        del self.number


    def mini_batch( self, int n_batch_size = 32, int n_window = 4, bint shuffle = True ):
        cdef int i, j, k
        cdef int max_word_length = 10
        cdef int cnt = 0
        cdef vector[string] words
        cdef vector[int] indices
        cdef vector[int] char_buff
        cdef vector[int] target
        cdef vector[vector[int]] char_idx

        cdef int unk = len( self.idx2word ) - 1
        cdef int eos = unk - 1
        cdef int bos = eos - 1

        candidate = numpy.arange( self.original.size() )
        if shuffle:
            numpy.random.shuffle( candidate )

        for i in candidate:
            with nogil:
                words = self.original[i]
                indices = self.numeric[i]
                for j in range( words.size() ):
                    if words[j].size() > max_word_length:
                        max_word_length = words[j].size()

                    for k in range( words[j].size() ):
                        char_buff.push_back( <int>words[j][k] )
                    char_idx.push_back( char_buff )

                    k = j - 1
                    while j - k <= n_window:
                        if k >= 0:
                            target.push_back( indices[k] )
                        else:
                            target.push_back( bos )
                        k -= 1

                    k = j + 1
                    while k - j <= n_window:
                        if k < words.size():
                            target.push_back( indices[k] )
                        else:
                            target.push_back( eos )
                        k += 1

                    char_buff.clear()
                    cnt += 1
                    if cnt == n_batch_size or i == self.original.size() - 1:
                        for k in range( char_idx.size() ):
                            while char_idx[k].size() < max_word_length:
                                char_idx[k].push_back( 0 )
                        with gil:
                            yield numpy.asarray( char_idx ), \
                                  numpy.asarray( target ).reshape([-1, n_window * 2])
                        char_idx.clear()
                        target.clear()
                        cnt = 0
                        max_word_length = 10


    def mini_batch_multi_thread( self, int n_batch_size = 32, 
                                 int n_window = 4, bint shuffle = True ):
        def prepare_mini_batch( batch_generator, batch_buffer ):
            for x in batch_generator:
                batch_buffer.put( x, True, None )
            batch_buffer.put( None, True, None )

        batch_buffer = Queue( maxsize = 128 )
        batch_generator = self.mini_batch( n_batch_size, n_window, shuffle )

        t = Thread( target = prepare_mini_batch, 
                    args = ( batch_generator, batch_buffer ) )
        t.daemon = True
        t.start()
        
        while True:
            next_batch = batch_buffer.get( True, None )
            if next_batch is not None:
                yield next_batch
            else:
                break



def word_batch( tokens, int n_batch_size = 32 ):
    cdef vector[string] words = tokens
    cdef vector[vector[int]] char_idx

    cdef int i, j
    cdef int max_word_length = 10
    cdef int cnt = 0
    cdef vector[int] char_buff

    for i in range( words.size() ):
        if words[i].size() > max_word_length:
            max_word_length = words[i].size()

        for j in range( words[i].size() ):
            char_buff.push_back( <int>words[i][j] )

        char_idx.push_back( char_buff )
        char_buff.clear()

        if (i + 1) % n_batch_size == 0 or i == words.size() - 1:
            for j in range( char_idx.size() ):
                while char_idx[j].size() < max_word_length:
                    char_idx[j].push_back( 0 )
            yield numpy.asarray( char_idx )
            max_word_length = 10
            char_idx.clear()

                
################################################################################

# logging.basicConfig( format = '%(asctime)s : %(levelname)s : %(message)s', 
#                      level= logging.INFO)

# bc = batch_constructor( 'reuters.txt' )
