import random

import numpy as np
from random import Random


def condensed_print(matrix):
    for i in matrix:
        for j in i:
            print(j, end='')
        print()


def print_all_forms():
    for piece in TetrisEnv.Pieces:
        print(piece + ":")
        print('---')
        condensed_print(TetrisEnv.Pieces[piece])
        print('#')
        condensed_print(np.rot90(TetrisEnv.Pieces[piece], axes=(1, 0)))
        print('#')
        condensed_print(np.rot90(TetrisEnv.Pieces[piece], 2, axes=(1, 0)))
        print('#')
        condensed_print(np.rot90(TetrisEnv.Pieces[piece], 3, axes=(1, 0)))
        print('---')
        print()


class TetrisEnv:
    SCORE_PIXEL = 1
    SCORE_SINGLE = 40 * 10
    SCORE_DOUBLE = 100 * 10
    SCORE_TRIPLE = 300 * 10
    SCORE_TETRIS = 1200 * 10
    MAX_TETRIS_ROWS = 20
    GAMEOVER_ROWS = 4
    TOTAL_ROWS = MAX_TETRIS_ROWS + GAMEOVER_ROWS
    MAX_TETRIS_COLS = 10
    GAMEOVER_PENALTY = -1000
    TETRIS_GRID = (TOTAL_ROWS, MAX_TETRIS_COLS)
    TETRIS_PIECES = ['O', 'I', 'S', 'Z', 'T', 'L', 'J']
    # Note, pieces are rotated clockwise
    Pieces = {'O': np.ones((2, 2), dtype=np.byte),
              'I': np.ones((4, 1), dtype=np.byte),
              'S': np.array([[0, 1, 1], [1, 1, 0]], dtype=np.byte),
              'Z': np.array([[1, 1, 0], [0, 1, 1]], dtype=np.byte),
              'T': np.array([[1, 1, 1], [0, 1, 0]], dtype=np.byte),
              'L': np.array([[1, 0], [1, 0], [1, 1]], dtype=np.byte),
              'J': np.array([[0, 1], [0, 1], [1, 1]], dtype=np.byte),
              }
    '''
    I:   S:      Z:      T:
      1      1 1    1 1     1 1 1
      1    1 1        1 1     1
      1
      1
    L:      J:      O:
      1        1      1 1
      1        1      1 1
      1 1    1 1
     last one is utf
    '''

    def __init__(self):
        self.RNG = Random()  # independent RNG
        self.default_seed = 17  # default seed is IT
        self.__restart()

    def __restart(self):
        self.RNG.seed(self.default_seed)
        self.board = np.zeros(self.TETRIS_GRID, dtype=np.byte)
        self.current_piece = self.RNG.choice(self.TETRIS_PIECES)
        self.next_piece = self.RNG.choice(self.TETRIS_PIECES)
        self.score = 0

    def __gen_next_piece(self):
        self.current_piece = self.next_piece
        self.next_piece = self.RNG.choice(self.TETRIS_PIECES)

    def set_seed(self, seed_value):
        self.default_seed = seed_value

    def get_status(self):
        return self.board.copy(), self.current_piece, self.next_piece

    # while can move down piece, move it down (note to restrict col to rotation max)
    # which is COLS-1 - (piece width in cur rotation -1) or cancel both -1s utf-8 #
    # check if move down, row++, if not, print piece on last row, col
    def __get_score(self, value):
        if value == 1:
            return TetrisEnv.SCORE_SINGLE
        if value == 2:
            return TetrisEnv.SCORE_DOUBLE
        if value == 3:
            return TetrisEnv.SCORE_TRIPLE
        if value == 4:
            return TetrisEnv.SCORE_TETRIS
        return 0

    def __collapse_rows(self, board):
        start_collapse = -1
        for row, i in zip(board, range(TetrisEnv.TOTAL_ROWS)):
            if np.sum(row) == TetrisEnv.MAX_TETRIS_COLS:
                start_collapse = i
                break
        if start_collapse == -1:
            return 0, board
        end_collapse = start_collapse + 1
        while end_collapse < TetrisEnv.TOTAL_ROWS:
            if np.sum(board[end_collapse]) == TetrisEnv.MAX_TETRIS_COLS:
                end_collapse += 1
            else:
                break
        new_board = np.delete(board, slice(start_collapse, end_collapse), axis=0)  # now we need to add them
        new_board = np.insert(new_board, slice(0, end_collapse - start_collapse), 0, axis=0)
        score = self.__get_score(end_collapse - start_collapse)

        return score, new_board

    def __game_over(self, test_board):
        return np.sum(test_board[:TetrisEnv.GAMEOVER_ROWS]) > 0

    def __play(self, col, rot_count):
        falling_piece = self.Pieces[self.current_piece]
        if rot_count > 0:
            falling_piece = np.rot90(falling_piece, rot_count, axes=(1, 0))
        p_dims = falling_piece.shape
        col = min(col, TetrisEnv.MAX_TETRIS_COLS - p_dims[1])
        max_row = TetrisEnv.TOTAL_ROWS - p_dims[0]
        chosen_row = 0
        while chosen_row < max_row:
            next_row = chosen_row + 1
            if np.sum(np.multiply(falling_piece,
                                  self.board[next_row:next_row + p_dims[0], col:col + p_dims[1]])) > 0:
                break
            chosen_row = next_row
        self.board[chosen_row:chosen_row + p_dims[0], col:col + p_dims[1]] |= falling_piece
        collapse_score, new_board = self.__collapse_rows(self.board)
        collapse_score += np.sum(falling_piece) * TetrisEnv.SCORE_PIXEL
        if self.__game_over(new_board):
            return TetrisEnv.GAMEOVER_PENALTY
        self.board = new_board
        return collapse_score

    # does not affect the class, tests a play of the game given a board and a piece b64 #
    def test_play(self, board_copy, piece_type, col, rot_count):
        falling_piece = self.Pieces[piece_type]
        if rot_count > 0:
            falling_piece = np.rot90(falling_piece, rot_count, axes=(1, 0))
        p_dims = falling_piece.shape
        col = min(col, TetrisEnv.MAX_TETRIS_COLS - p_dims[1])
        max_row = TetrisEnv.TOTAL_ROWS - p_dims[0]
        chosen_row = 0
        while chosen_row < max_row:
            next_row = chosen_row + 1
            if np.sum(np.multiply(falling_piece,
                                  board_copy[next_row:next_row + p_dims[0], col:col + p_dims[1]])) > 0:
                break
            chosen_row = next_row
        board_copy[chosen_row:chosen_row + p_dims[0], col:col + p_dims[1]] |= falling_piece
        collapse_score, board_copy = self.__collapse_rows(board_copy)
        collapse_score += np.sum(falling_piece) * TetrisEnv.SCORE_PIXEL
        if self.__game_over(board_copy):
            return TetrisEnv.GAMEOVER_PENALTY, board_copy
        return collapse_score, board_copy

    def __calc_rank_n_rot(self, scoring_function, genetic_params, col):
        # should return rank score and rotation a pair (rank,rot), rot is from 0 to 3
        return scoring_function(self, genetic_params, col)

    def __get_lose_msg(self):
        # if understood, send to owner
        lose_msg = b'TFVMISBfIFlPVSBMT1NFIQrilZbilKTilKTilLzilZHilaLilaLilaLilaLilaLilaLilaPilaLilaLilaPilaLilaLilaLilazilazilazilazilazilazilaPilaPilaLilaLilaLilaLilaLilaLilaPilaLilazilazilazilazilazilaPilaPilaPilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaIK4pSk4pWc4pWc4pSC4pSU4pSU4pWZ4pWZ4pWc4pWc4pWc4pWZ4pWZ4pWZ4pWZ4pWc4pWc4pWi4pWi4pWi4pWi4pWr4pWs4pWj4pWj4pWj4pWj4pWi4pWi4pWi4pWi4pWi4pWc4pWc4pWc4pWc4pWc4pWc4pWc4pWi4pWj4pWj4pWc4pWc4pWc4pWc4pWc4pSk4pSC4pSC4pSC4pSC4pWc4pWc4pWc4pWR4pWc4pWR4pWi4pWiCuKVnOKUguKUlCAgICAgICAgICAgIOKUguKUguKUguKVkeKVouKVouKVouKVouKVouKVouKVo+KVouKVouKVouKVouKVnOKVnOKUguKUguKUguKUguKUlCAgIOKUlOKUlOKUlOKUlOKUlCDilJTilZnilKTilKTilKTilKTilZzilZzilZzilZzilZzilZzilZzilZwK4pSC4pSU4pSM4pSM4pSM4pWT4pWT4pWT4pWT4pWT4pWT4pWT4pWT4pWTICAg4pSU4pWZ4pWi4pWi4pWi4pWR4pWi4pWi4pWj4pWs4pWi4pWR4pWc4pSC4pSC4pSC4pSC4pSUICAgICAgICDilZPilZbilZbilZbilZbilZbilZbilKTilKTilKTilKTilKTilKTilZzilKTilKTilZwK4pSC4pSC4pWT4pWR4pWi4pWi4pWi4pWi4pWj4pWj4pWj4pWi4pWi4pWi4pWj4pWj4pWW4pWW4pSM4pSU4pWZ4pWc4pWc4pWZ4pWi4pWi4pWj4pWi4pWi4pWi4pSk4pSC4pSC4pSC4pWW4pWW4pWW4pWW4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWj4pWj4pWj4pWi4pWi4pWi4pWi4pSk4pWc4pSk4pWc4pWc4pWc4pWcCuKUguKUlCAgICAgICAg4pSU4pWZ4pWc4pWc4pSC4pWZ4pWc4pWc4pWi4pWW4pWW4pWW4pWW4pWW4pWi4pWr4pWs4pWj4pWi4pWi4pWi4pWi4pWW4pSk4pSC4pSC4pWc4pWc4pWc4pWc4pWc4pWc4pWc4pWc4pWZ4pWZ4pWZ4pWZ4pWZ4pWc4pWc4pWc4pWc4pWc4pWi4pWi4pWR4pSk4pSk4pSkCuKVluKUkCAgIOKVk+KVk+KVluKVluKVluKVluKVluKVluKVluKVouKVouKVouKVouKVouKVouKVouKVouKVouKVouKVouKVouKVo+KVo+KVo+KVouKVouKVouKVouKVouKVluKVouKVouKUpOKUguKUguKUlCAgICAg4pSM4pSMICAgICAg4pSM4pSC4pSC4pSC4pWc4pWcCuKVouKVluKVnOKUpOKUguKUguKUguKVnOKVnOKVnOKVnOKVouKVouKVouKVnOKVouKVouKVouKVouKVouKVouKVouKVouKVouKVouKVouKVo+KVouKVouKVouKVouKVouKVouKVouKVouKVouKVouKVouKVluKUpOKUvOKVouKVouKVrOKVrOKVrOKVo+KVo+KVouKVouKVouKVouKVo+KVouKVouKVluKVluKVluKVouKVogrilaLilaLilZbilZbilZbilZbilILilILilIzilZPilZbilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaPilaPilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilZbilILilILilILilZnilZnilZnilZzilZzilaLilaLilaLilaLilaLilaLilaLilaLilaLilaIK4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWj4pWj4pWi4pWi4pWi4pWi4pWi4pWc4pWc4pWc4pWc4pWR4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWj4pWi4pWi4pWi4pWi4pWi4pWi4pWj4pWi4pWi4pWi4pWi4pWi4pWj4pWi4pWi4pWi4pWi4pWi4pWiCuKVnOKVouKVouKVouKVouKVouKVouKVouKVo+KVo+KVouKVouKVouKVouKVouKVouKVnOKVnOKVnOKVnOKVluKVouKVouKVouKVouKVouKVouKVouKVouKVnOKVnOKVnOKVouKVouKVouKVnOKVnOKVq+KVrOKVrOKVrOKVo+KVouKVouKVouKVouKVouKVouKVouKVrOKVrOKVrOKVrOKVo+KVo+KVouKVouKVouKVouKVogrilZHilaLilaLilaLilaLilaLilaLilaLilaPilaPilaLilaLilaLilZzilZzilaLilZHilKTilILilZHilaLilaLilaLilaLilaPilaLilKTilKTilKTilILilILilZbilZHilaLilaLilZbilZbilKTilZzilZzilavilazilazilazilazilazilazilazilazilazilazilazilaPilaPilaPilaLilaPilaLilaLilaIK4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWc4pSC4pSC4pWR4pWc4pWc4pWc4pWc4pWc4pWi4pWi4pWc4pWc4pWc4pWc4pWc4pWc4pSk4pWW4pWR4pWi4pWi4pWi4pWi4pWc4pWc4pWZ4pWi4pWi4pWW4pWZ4pWZ4pWi4pWr4pWs4pWs4pWs4pWs4pWs4pWj4pWj4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWiCuKVnOKVnOKVnOKVnOKVouKVouKVouKVouKVouKVnOKVnOKUguKVluKVouKVnOKVnOKVmeKVmeKVnOKUpOKUpOKUpOKUguKUguKUguKUguKVnOKVnOKVnOKVnOKVnOKVqOKVqOKVnOKVnOKVnOKUguKUguKVkeKVouKVo+KVouKUpOKVnOKVnOKVnOKVnOKVnOKVnOKVouKVouKVouKVouKVouKVouKVouKVouKVouKVouKVogrilILilILilILilILilZzilZzilaLilZzilZzilILilILilZPilZzilJggICAgICAgIOKUlOKUlOKUlCAgICAgICAgIOKUjCAg4pSC4pWc4pSC4pSC4pSC4pWc4pSk4pSk4pWc4pWc4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWc4pWi4pWi4pWR4pWcCuKUguKUguKUguKUguKUguKUguKUguKUguKUguKUguKUguKUmCAgICAgICAgICAg4pSM4pWT4pWT4pSQICDilJTilJTilJTilJTilJQgIOKUlOKUguKUguKUlOKUlOKUlOKUlOKUguKUguKUguKUguKUguKUguKVnOKVnOKVnOKVnOKVnOKVnOKVnOKVnOKVnOKVnOKUggrilILilILilILilILilILilILilILilILilILilILilJQgICAgICAgICAgICAg4pSC4pSCICAgICDilIwgICAgIOKUguKUgiAgICAg4pSU4pSC4pSC4pSC4pSC4pSC4pSC4pWc4pWc4pWc4pWc4pSk4pSk4pSC4pSC4pSCCuKUguKUguKUguKUguKUguKUguKUguKUguKUgiAgICAgICAgICAgICAg4pSM4pWT4pWW4pSQICAgIOKUguKUgiAgICAg4pSUICAgICAgICAg4pSU4pSC4pSC4pSC4pSC4pSC4pSk4pSk4pSC4pSC4pSC4pSkCuKUguKUguKUguKUguKUguKUguKUguKUpOKUmCAgICAgICAgICAgICDilJTilZnilZzilZzilZzilKTilJAg4pSM4pSC4pSM4pSMICAg4pSM4pSM4pSQ4pSMICAgICAgICAg4pWZ4pWc4pSC4pSC4pSC4pSk4pSk4pSk4pSk4pSkCuKUguKUguKUguKUguKUguKUguKUguKUpOKUkCAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICDilJTilJTilJjilJAgICAgIOKUlOKVnOKVnOKVnOKUpOKUpOKUpOKUpArilILilILilILilILilILilILilILilZHilZbilZbilJDilZPilILilIIgICDilJTilavilazilazilaPilIAgICAgICAgICAgICAgICAgICAgICAgICAg4pSU4pWW4pWW4pWW4pSC4pSC4pWW4pSk4pSk4pWc4pSk4pSCCuKUguKUguKUguKUguKUguKUguKUguKVnOKVnOKVouKVo+KUpOKUguKUguKVkeKVouKVliAgICAgICDilZPilZPilZMgICAgICAgIOKVk+KVk+KVk+KVluKVluKVluKVluKVluKVluKVluKUkCAg4pWT4pWW4pSC4pSC4pSC4pSC4pWR4pWc4pWc4pSk4pSC4pSCCuKUguKUguKUguKUguKUguKUguKUguKUguKUguKVmeKVouKUpOKUguKUguKVmeKVouKVouKUpOKVluKVluKVpeKVpeKVo+KVo+KVouKVouKVouKVrOKVrOKVrOKVrOKVrOKVrOKVrOKVo+KVo+KVo+KVouKVouKVouKVnOKVnOKVnOKVnOKUguKUguKVk+KVkeKVouKVnOKUguKUguKUguKUguKVnOKVnOKUpOKUguKUguKUggrilILilILilILilILilILilILilILilILilILilZzilZzilKTilKTilZbilILilZHilaLilKTilILilILilZnilaLilaLilaPilaPilaPilaLilaPilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaPilaLilZzilKTilILilILilZbilaLilaPilZzilZzilKTilILilILilILilKTilZzilKTilKTilILilILilIIK4pSC4pSC4pSC4pSC4pSC4pSC4pSC4pWR4pWW4pSC4pSC4pSk4pSk4pSk4pSk4pSk4pWR4pSk4pSC4pSC4pSC4pSC4pWR4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWj4pWi4pWc4pWc4pWc4pWc4pWc4pSC4pSC4pSC4pWT4pWc4pWi4pWi4pWi4pWi4pWi4pWi4pSk4pSk4pSk4pSk4pWc4pWc4pSk4pSC4pSC4pSC4pSCCuKUguKUguKUguKUguKUguKUguKUguKVkeKVouKVluKUguKVmeKVkeKUpOKUpOKVnOKVnOKVnOKUpOKUguKUguKUguKUpOKVnOKUpOKUpOKVnOKVnOKVnOKVnOKVnOKVnOKUguKVk+KVk+KVq+KVrOKVrOKVmeKVnOKVnOKVnOKVnOKVkeKVouKVouKVouKVouKVouKVnOKUpOKUpOKUpOKVnOKUpOKUpOKUpOKUguKUguKUggrilILilILilILilILilILilILilILilZHilaLilaLilKTilILilZzilZzilKTilILilILilILilKTilZbilILilILilZzilZHilZHilZHilZHilZHilZzilKTilZbilZbilaLilaLilaLilaPilZzilZzilZbilZbilZbilZbilZbilaLilaLilaLilaLilaLilZzilZzilKTilZzilZzilZzilZzilKTilILilILilILilZE='
        return lose_msg

    def run(self, scoring_function, genetic_params, num_of_iters, return_trace):
        self.__restart()
        # no trace
        if not return_trace:
            for it in range(num_of_iters):
                rates = []
                rotations = []
                for c in range(TetrisEnv.MAX_TETRIS_COLS):
                    r1, r2 = self.__calc_rank_n_rot(scoring_function, genetic_params, c)
                    rates.append(r1)
                    rotations.append(r2)
                pos_to_play = rates.index(max(rates))  # plays first max found
                rot_to_play = rotations[pos_to_play]
                play_score = self.__play(pos_to_play, rot_to_play)
                self.score += play_score
                self.__gen_next_piece()
                if play_score < 0:
                    return self.score, self.board, self.__get_lose_msg()
            return self.score, self.board, ""
        else:  # we want to trace
            board_states = []
            ratings_n_rotations = []
            pieces_got = []
            # board_states.append(self.board.copy())
            for it in range(num_of_iters):
                rates = []
                rotations = []
                pieces_got.append(self.current_piece)
                for c in range(TetrisEnv.MAX_TETRIS_COLS):
                    r1, r2 = self.__calc_rank_n_rot(scoring_function, genetic_params, c)
                    rates.append(r1)
                    rotations.append(r2)
                ratings_n_rotations.append(list(zip(rates, rotations)))
                pos_to_play = rates.index(max(rates))  # plays first max found
                rot_to_play = rotations[pos_to_play]
                play_score = self.__play(pos_to_play, rot_to_play)
                self.score += play_score
                self.__gen_next_piece()
                board_states.append(self.board.copy())
                if play_score < 0:
                    return self.score, board_states, ratings_n_rotations, pieces_got, self.__get_lose_msg()
            return self.score, board_states, ratings_n_rotations, pieces_got, ""
        # don't really feel like removing redundancy, cleaning code


# max gain + random
def random_scoring_function(tetris_env: TetrisEnv, gen_params, col):
    board, piece, next_piece = tetris_env.get_status()  # add type hinting
    scores = []
    for i in range(4):
        score, tmp_board = tetris_env.test_play(board, piece, col, i)
        if score < 0:
            scores.append([score * gen_params[0], i])
            continue
        tmp_scores = []
        for t in range(tetris_env.MAX_TETRIS_COLS):
            for j in range(4):
                score2, _ = tetris_env.test_play(tmp_board, next_piece, t, j)
                tmp_scores.append(score2 * gen_params[1])
        max_score2 = max(tmp_scores)
        if max_score2 >= 0:
            score += max_score2
        scores.append([score, i])
    for i in range(4):
        scores[i][0] *= random.randint(1, gen_params[2])
    val = max(scores, key=lambda item: item[0])  # need to store it first or it iterates
    # print(val)
    return val[0], val[1]


def print_stats(use_visuals_in_trace_p, states_p, pieces_p, sleep_time_p):
    vision = BoardVision()
    if use_visuals_in_trace_p:

        for state, piece in zip(states_p, pieces_p):
            vision.update_board(state)
            # print("piece")
            # condensed_print(piece)
            # print('-----')
            time.sleep(sleep_time_p)
        time.sleep(2)
        vision.close()
    else:
        for state, piece in zip(states_p, pieces_p):
            print("board")
            condensed_print(state)
            print("piece")
            condensed_print(piece)
            print('-----')


'''
#1 double holes (a hole with 2 gaps is worse (like the Z top)
#2 one hole gap
#3 line continuation score (have more lines with less empty blocks is better)
    each line score = number of pieces in it ^2 * constant
#4 Max height, max height gets penalized after being 4 and above
#X5 to prefer making 2-4 collapses, it would need to NOT score unless it has more
    or the height situation is getting risky
#6 10-index points (to make it stack stuff on the left first if there is no better
    play 
# A difficult un intuative solution would be a pattern block fitter added to the
    equation (like valleys being reserved for L and J and I
    
# started at 6 am prolly // pause on 6:30 need to study genetic algorithm
    to apply a good evolution
# chose not to apply genetic algorithms, fixed a pass by reference issue
    code worked fine after a few tweaks to the chromosome
# added other usable rating functions


# we should add to penalize valleys 0.2

'''


# region my functions
def count_holes_t1(board):
    # count top rows bigger than bottom
    return np.sum(board[:-1] > board[1:])


def count_holes_t2(board):
    # the top 4 lines are essentially useless
    # the 1:-1 here because you are comparing a point with the 2 points above it
    # the second line is not factored in this calculation and both matrices need to have
    # same size for it to have meaning
    return np.sum(np.multiply(board[1:-1] > board[2:], board[:-2] > board[2:]))


def line_continuation(board):
    return np.sum(np.square(np.sum(board, axis=1)))


def max_height(board):
    h = 0
    vals = np.sum(board, axis=1)
    for i in range(vals.shape[0]):
        if vals[i] > 0:
            h = 24 - i
            break
    h = max(0, h - 4)  # ignore first 4 rows
    return h


def left_best(col):  # give bias to putting things on the left when there is no other reason
    return TetrisEnv.MAX_TETRIS_COLS - col


# endregion

#
#
#
#
#

# region extra functions (not my ideas):
def get_peaks(area):
    peaks = np.array([])
    for col in range(area.shape[1]):
        if 1 in area[:, col]:
            p = area.shape[0] - np.argmax(area[:, col], axis=0)
            peaks = np.append(peaks, p)
        else:
            peaks = np.append(peaks, 0)
    return peaks


def aggregated_height(peaks):
    return np.sum(peaks)


def highest_peak(peaks):
    return np.max(peaks)


def get_bumpiness(peaks):
    s = 0
    for i in range(TetrisEnv.MAX_TETRIS_COLS - 1):
        s += np.abs(peaks[i] - peaks[i + 1])
    return s


def count_holes_n_cols_with_them(board):  # used for two things
    # count top rows bigger than bottom
    temp = board[:-1] > board[1:]
    return np.sum(temp), np.count_nonzero(np.sum(temp, axis=0))


def get_wells(peaks):
    wells = []
    for i in range(len(peaks)):
        if i == 0:
            w = peaks[1] - peaks[0]
            w = w if w > 0 else 0
            wells.append(w)
        elif i == len(peaks) - 1:
            w = peaks[-2] - peaks[-1]
            w = w if w > 0 else 0
            wells.append(w)
        else:
            w1 = peaks[i - 1] - peaks[i]
            w2 = peaks[i + 1] - peaks[i]
            w1 = w1 if w1 > 0 else 0
            w2 = w2 if w2 > 0 else 0
            w = w1 if w1 >= w2 else w2
            wells.append(w)
    return wells


def get_deepest_well(peaks):
    wells = get_wells(peaks)
    return np.max(wells)


# endregion

# ratings used and their corresponding winning chromosome (not necessarily optimal)

# ratings.append(count_holes_t1(tmp_board))
# ratings.append(count_holes_t2(tmp_board))
# ratings.append(line_continuation(tmp_board))
# ratings.append(max_height(tmp_board))
# ratings.append(left_best(col))
# turns out this works, the copy() part was making issues

eternal_chromo = [-10, -5, 0.5, 1, 1]
# now that I think about it, the +ve height may have been unintentional...

# this was kind of brute forced to avoid genetic algorithm unnecessities (for me)

# others:
# ratings.append(aggregated_height(board_peaks))
# holes, cols_with_holes = count_holes_n_cols_with_them(tmp_board)
# ratings.append(holes)
# ratings.append(cols_with_holes)
# ratings.append(get_deepest_well(board_peaks))
# ratings.append(get_bumpiness(board_peaks))
# took 45 minutes to get and some time to find the error

# eternal_chromo = [-3,  -4, -8, -4, -0.4]

# unused, but generally faster versions with less lines (than those in the website)
# remember python is a scripting language, being fast in it requires writing faster codes
#   and less lines, as lines require time to translate.
# looping is slower than slicing and comparing (even though he does limit the loop by
#   relying on max peaks, he still needs time to calculate those peaks and
#   in larger sizes more time to iterate loops than the numpy would need )
# The iteration looping in python is slower than built in iteration

def row_transition(board):
    return np.sum(board[:, :-1] != board[:, 1:])


def col_transition(board):  # similar to holes a bit
    return np.sum(board[:-1] != board[1:])


def eternal(tetris_env: TetrisEnv, gen_params, col):
    board, piece, next_piece = tetris_env.get_status()  # add type hinting
    scores = []
    for i in range(4):
        score, tmp_board = tetris_env.test_play(board.copy(), piece, col, i)
        # condensed_print(tmp_board)
        # print("test")
        # score *= gen_params[5]
        if score < 0:
            scores.append([score, i])
            continue
        ratings = []


        ratings.append(count_holes_t1(tmp_board))
        ratings.append(count_holes_t2(tmp_board))
        ratings.append(line_continuation(tmp_board))
        ratings.append(max_height(tmp_board))
        ratings.append(left_best(col))




        # other ratings : (ref : https://towardsdatascience.com/beating-the-world-record-in-tetris-gb-with-genetics-algorithm-6c0b2f5ace9b?gi=55fe5bdcbb8a)

        # board_peaks = get_peaks(tmp_board)
        # ratings.append(aggregated_height(board_peaks))
        # holes, cols_with_holes = count_holes_n_cols_with_them(tmp_board)
        # ratings.append(holes)
        # ratings.append(cols_with_holes)
        # ratings.append(get_deepest_well(board_peaks))
        # ratings.append(get_bumpiness(board_peaks))
        # no pits yet, no transitions, the above was enough as a secondary test

        # print(ratings)

        for g in range(len(gen_params)):
            score += ratings[g] * gen_params[g]

        # No need for extra calculations

        # tmp_scores = rate_1*
        # for t in range(tetris_env.MAX_TETRIS_COLS):
        #     for j in range(4):
        #         score2, _ = tetris_env.test_play(tmp_board, next_piece, t, j)
        #         tmp_scores.append(score2 * gen_params[1])
        # max_score2 = max(tmp_scores)
        # if max_score2 >= 0:
        #     score += max_score2
        scores.append([score, i])

    val = max(scores, key=lambda item: item[0])  # need to store it first or it iterates
    # print(val)
    return val[0], val[1]


if __name__ == "__main__":
    use_visuals_in_trace = True
    sleep_time = 0.1
    # just one chromosome in the population
    # one_chromo_competent = [-4, -1, 2,3]
    from Visor import BoardVision
    import time

    # print_all_forms()
    env = TetrisEnv()
    # total_score, states, rate_rot, pieces, msg = env.run(
    #     random_scoring_function, one_chromo_rando, 100, True)
    total_score, states, rate_rot, pieces, msg = env.run(
        eternal, eternal_chromo, 600, True)
    # after running your iterations (which should be at least 500 for each chromosome)
    # you can evolve your new chromosomes from the best after you test all chromosomes here
    # print("Ratings and rotations")
    # for rr in rate_rot:
    #     print(rr)
    # print('----')
    # print('steps', len(rate_rot))

    print(total_score)
    print(msg)
    print_stats(use_visuals_in_trace, states, pieces, sleep_time)
    # env.set_seed(5132)
    # total_score, states, rate_rot, pieces, msg = env.run(
    #     eternal, one_chromo_rando, 100, True)
    # print("Ratings and rotations")
    # for rr in rate_rot:
    #     print(rr)
    # print('----')
    # print(total_score)
    # print(msg)
    # print_stats(use_visuals_in_trace, states, pieces, sleep_time)

# use log instead of printing for traces
# use smaller fonts for the message
