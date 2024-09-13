from functools import cached_property, cache
import numpy as np

from .utilities import _calculate_bottom_mask, _get_column_order, _top_mask_cols, _bottom_mask_cols, _column_masks, \
    _popcount, _compute_winning_position, _can_play, _is_winning_move, _key3

class Board:
    
    width = 7
    height = 6
    
    min_score = -(width * height) // 2 + 3
    max_score = (width * height + 1) // 2 - 3

    max_moves = width * height
    
    assert width * (height + 1) <= 64, "Width * (Height + 1) too large."

    bottom_mask = np.uint64(_calculate_bottom_mask(width, height))
    board_mask = bottom_mask * ((np.uint64(1) << height) - 1)

    column_order = _get_column_order(width)
    top_mask_col = _top_mask_cols(width, height)
    bottom_mask_col = _bottom_mask_cols(width, height)
    column_mask = _column_masks(width, height)

    boards_ = {}

    def __new__(cls, current_position=0, mask=0):
        position_id = (current_position, mask)
        if position_id not in cls.boards_:
            cls.boards_[position_id] = super().__new__(cls)
        return cls.boards_[position_id]
    
    def __init__(self, current_position=0, mask=0):
        self.current_position = np.uint64(current_position)
        self.mask = np.uint64(mask)
    
    @cached_property
    def moves(self):
        return _popcount(self.mask)
    
    @cached_property
    def key(self):
        return self.current_position + self.mask
    
    def __hash__(self):
        return hash(self.key)

    def __repr__(self):
        position_array = self.seq_to_array(self.int_to_seq(self.current_position))[1:]
        mask_array = self.seq_to_array(self.int_to_seq(self.mask))[1:]
        
        next_player = 'o' if self.moves % 2 == 0 else 'x'
        board = f'{next_player} - moves next'
        for row_mask, row_position in zip(mask_array, position_array):
            board += '\n'
            for mask_bit, position_bit in zip(row_mask, row_position):
                if mask_bit == '1':
                    if position_bit == '1':
                        board += 'x' if self.moves % 2 == 1 else 'o'
                    else:
                        board += 'o'if self.moves % 2 == 1 else 'x'
                else:
                    board += '.'
        return board
    
    @classmethod
    def animate(cls, seq, fps=1):
        import time
        from IPython.display import display, clear_output

        board = cls()
        for col in seq:
            clear_output(wait=True)
            board = board.play(col)
            display(board)
            time.sleep(1/fps)
    
    @staticmethod
    def int_to_seq(ints):
        bit_repr = ('{'+f':0{Board.width*(Board.height+1)}b'+'}')
        return bit_repr.format(ints)
    
    @staticmethod
    def seq_to_array(seq):
        return np.array([[char for char in seq[i:i+(Board.height+1)]] for i in range(0, len(seq), (Board.height+1))]).T[:, ::-1]
    
    @staticmethod
    def array_to_board(array):
        board = ''
        for row in array:
            for char in row:
                board += char
            board += '\n'
        return board
    
    @staticmethod
    def int_to_board(ints):
        seq = Board.int_to_seq(ints)
        array = Board.seq_to_array(seq)
        board = Board.array_to_board(array)
        return board
    
    def play(self, seq):
        position = self
        for col in seq:
            col = int(col) - 1
            if (col < 0) or (col >= Board.width) or \
              (not position.can_play(col)) or position.is_winning_move(col):
                return position
            position = position.play_col(col)
        return position
    
    @cache
    def play_col(self, col):
        return self.play_move((self.mask + self.bottom_mask_col[col]) & self.column_mask[col])
    
    def play_move(self, move):
        current_position = self.current_position ^ self.mask
        mask = self.mask | move
        return Board(current_position=current_position, mask=mask)

    @cached_property
    def can_win_next(self):
        return self.winning_position & self.possible
    
    @cached_property
    def possible_non_losing_moves(self):
        assert not self.can_win_next
        possible_mask = self.possible
        opponent_win = self.opponent_winning_position

        forced_moves = possible_mask & opponent_win
        if forced_moves:
            if forced_moves & (forced_moves - 1):
                return 0
            else:
                possible_mask = forced_moves
        
        return possible_mask & ~(opponent_win >> 1)
    
    def compute_winning_position(self, position, mask):
        return np.uint64(_compute_winning_position(position, mask, self.height, self.board_mask))

    def move_score(self, move):
        return _popcount(self.compute_winning_position(self.current_position | move, self.mask))
    
    def can_play(self, col):
        return _can_play(self.mask, self.top_mask_col[col])
    
    def is_winning_move(self, col):
        return _is_winning_move(self.winning_position, self.possible, self.column_mask[col])
    
    @cached_property
    def winning_position(self):
        return self.compute_winning_position(self.current_position, self.mask)
    
    @cached_property
    def opponent_winning_position(self):
        return self.compute_winning_position(self.current_position ^ self.mask, self.mask)
    
    @cached_property
    def possible(self):
        return (self.mask + self.bottom_mask) & self.board_mask
    
    @cached_property
    def key3(self):
        return _key3(self.mask, self.current_position, self.height, self.width)
