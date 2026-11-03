import numpy as np 

# ---------------------------------------------------------------- COCO run-length masks (track_Kaden masks.json)
def rle_counts(s):
    '''COCO compressed RLE string -> run lengths (port of pycocotools rleFrString)'''
    if isinstance(s, bytes): s = s.decode('ascii')         # the efish4-track .pkl stores counts as bytes
    cnts, p = [], 0
    while p < len(s):
        x, k, more = 0, 0, 1
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1f) << 5 * k
            more = c & 0x20
            p += 1; k += 1
            if not more and (c & 0x10): x |= -1 << 5 * k
        if len(cnts) > 2: x += cnts[-2]
        cnts.append(x)
    return cnts


def rle_decode(rle):
    '''{'size': [h, w], 'counts': str} -> (h, w) bool mask, indexed [row = y, col = x]'''
    h, w = rle['size']
    cnts = np.asarray(rle_counts(rle['counts']), np.int64)
    vals = np.zeros(len(cnts), bool); vals[1::2] = True               # runs alternate 0, 1, 0, ... starting with 0
    flat = np.repeat(vals, cnts)
    return np.pad(flat, (0, h * w - len(flat))).reshape((h, w), order='F')   # COCO is column-major
