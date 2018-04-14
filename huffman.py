"""
Source :
https://rosettacode.org/wiki/Huffman_coding#Python
"""
from heapq import heappush, heappop, heapify
from collections import defaultdict

import numpy as np
import tensorflow as tf

def get_mapping(val, freq):
    """Huffman encode the given dict mapping symbols to weights"""
    heap = [[wt, [sym, ""]] for sym, wt in zip(val, freq)]
    heapify(heap)
    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

def get_compressed_size(src, mapping):
    """Return the compressed object size in bytes"""
    size = 0
    for val, code in mapping:
        size += np.sum((src == val) * len(code))
    return (size // 8)+1
 
def get_size(src):
    """Return the original object size in bytes"""
    return src.size * src.dtype.itemsize

def huffman_coding(src):
    """
    Return (mapping, original_size, compressed_size)
    with mapping a list of [symbol, code] 
    """
    val, freq = np.unique(src.ravel(),return_counts = True)

    mapping = get_mapping(val, freq)

    return mapping, get_size(src), get_compressed_size(src, mapping)

def main():
    # Random input
    src = np.random.random_integers(0,58,(5,5,96))

    # Huffman Encoding
    mapping, original_size, compressed_size = huffman_coding(src)

    # Print some results
    print("Symbol\tHuffman Code")
    for p in mapping:
        print("{}\t{}".format(p[0], p[1]))

    print("Original size: {}".format(original_size))
    print("Compressed size: {}".format(compressed_size))

if __name__ == '__main__':
    main()
