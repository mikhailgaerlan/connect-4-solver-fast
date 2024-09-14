from .board import Board

__all__ = ['solve']

def solve(board, weak=False):
    if board.can_win_next:
        return (Board.max_moves + 1 - board.moves) // 2
    min_score = -((Board.max_moves - board.moves) // 2)
    max_score = (Board.max_moves + 1 - board.moves) // 2
    if weak:
        min_score = -1
        max_score = 1
    
    while min_score < max_score:
        med_score = min_score + (max_score - min_score) // 2
        if med_score <= 0 and min_score // 2 < med_score: med_score = min_score // 2
        elif med_score >= 0 and max_score // 2 > med_score: med_score = max_score // 2
        r = negamax(board, med_score, med_score + 1)
        if r <= med_score: max_score = r
        else: min_score = r
    return min_score

NEGAMAX_CACHE = {}
NODE_COUNT = 0
def negamax(board, alpha, beta):
    assert alpha < beta
    assert not board.can_win_next

    global NODE_COUNT
    NODE_COUNT += 1
    
    if board.possible_non_losing_moves == 0:
        return -((Board.max_moves - board.moves) // 2)
    
    if board.moves >= Board.max_moves - 2:
        return 0
    
    min_score = -((Board.max_moves - 2 - board.moves) // 2)
    if alpha < min_score:
        alpha = min_score
        if alpha >= beta: return alpha
    
    max_score = (Board.max_moves - 1 - board.moves) // 2
    if beta > max_score:
        beta = max_score
        if alpha >= beta: return beta
    
    bound = NEGAMAX_CACHE.get(board.key, 0)
    if bound:
        if bound > Board.max_score - Board.min_score + 1:
            min_score = bound + 2 * Board.min_score - Board.max_score - 2
            if alpha < min_score:
                alpha = min_score
                if alpha >= beta: return alpha
        else:
            max_score = bound + Board.min_score - 1
            if beta > max_score:
                beta = max_score
                if alpha >= beta: return beta
    
    moves = MoveSorter()
    for col in Board.column_order[::-1]:
        move = board.possible_non_losing_moves & board.column_mask[col]
        if move: moves.insert_move(move, board.move_score(move))
    
    while moves:
        next_move = moves.pop().move
        board2 = board.play_move(next_move)
        score = -negamax(board2, -beta, -alpha)

        if score >= beta:
            NEGAMAX_CACHE[board.key] = score + Board.max_score - 2 * Board.min_score + 2
            return score
        if score > alpha: alpha = score
    
    NEGAMAX_CACHE[board.key] = alpha - Board.min_score + 1
    return alpha

class MoveSorter(list):

    def insert_move(self, move, score):
        new_score = MoveScore(move, score)
        for i, old_score in enumerate(self):
            if new_score.score < old_score.score:
                self.insert(i, new_score)
                return self
        self.append(new_score)
        return self

class MoveScore:

    def __init__(self, move, score):
        self.move = move
        self.score = score
    
    def __repr__(self):
        return f'Move(move={self.move}, score={self.score})'