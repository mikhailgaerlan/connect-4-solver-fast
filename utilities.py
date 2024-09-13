import numpy as np
from numba import njit

def _get_column_order(width):
    return [(width // 2 + (1 - 2 * (i % 2)) * (i + 1) // 2) for i in range(width)]

@njit
def _calculate_bottom_mask(width, height):
    if width > 0:
        return _calculate_bottom_mask(width-1, height) | (np.uint64(1) << ((width - 1) * (height + 1)))
    return 0

def _top_mask_cols(width, height):
    return [1 << ((height - 1) + col * (height + 1)) for col in range(width)]

def _bottom_mask_cols(width, height):
    return [1 << col * (height + 1) for col in range(width)]

def _column_masks(width, height):
    return [((1 << height) - 1) << col * (height + 1) for col in range(width)]

@njit
def _popcount(m):
    n = 0
    while m != 0:
        m &= m - np.uint64(1)
        n += 1
    return n

@njit
def _can_play(mask, top_mask):
    return (mask & top_mask) == 0

@njit
def _is_winning_move(winning_position, possible, col_mask):
    return winning_position & possible & col_mask

@njit
def _compute_winning_position(position, mask, height, board_mask):

    r = (position << 1) & (position << 2) & (position << 3)

    for h in [height+1, height, height+2]:
        p = (position << h) & (position << 2 * h)
        r |= p & (position << 3 * h)
        r |= p & (position >> h)
        p = (position >> h) & (position >> 2 * h)
        r |= p & (position << h)
        r |= p & (position >> 3 * h)

    return r & (board_mask ^ mask)

@njit
def _partial_key3(mask, current_position, height, key, col):
    pos = np.uint64(1) << (col * (height + 1))
    while pos & mask != 0:
        key *= 3
        if pos & current_position: key += 1
        else: key += 2
        pos <<= 1
    key *= 3
    return key

@njit
def _key3(mask, current_position, height, width):
    key_forward = np.uint64(0)
    for i in range(width):
        key_forward = _partial_key3(mask, current_position, height, key_forward, i)
    
    key_reverse = 0
    for i in range(width, 0, -1):
        key_reverse = _partial_key3(mask, current_position, height, key_reverse, i)
    
    return key_forward // 3 if key_forward < key_reverse else key_reverse // 3
