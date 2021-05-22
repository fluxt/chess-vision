import base64
import sys
import os

import chess
import chess.engine
import numpy as np
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import pyperclip
from joblib import dump, load
from PIL import Image, ImageGrab

import vision
import visualizer

clf = load("models/clf.joblib")

def predictBoard(tiles, is_white):
    X = tiles.reshape([64, 1024])
    y = clf.predict(X)
    if not is_white:
        y = np.flip(y)
    board = chess.Board()
    board.clear_board()
    for sqi, psymbol in enumerate(y):
        if psymbol != " ":
            board.set_piece_at(sqi, chess.Piece.from_symbol(psymbol))
    return board

stockfish_dir = "engines/stockfish_14_x64_avx2"

def main():
    is_white = True
    pygame.display.set_caption("White" if is_white else "Black")
    board = chess.Board()

    engine_allie = chess.engine.SimpleEngine.popen_uci(stockfish_dir)
    engine_allie_board = None
    engine_allie_result = None
    engine_enemy = chess.engine.SimpleEngine.popen_uci(stockfish_dir)
    engine_enemy_board = None
    engine_enemy_result = None

    def quit():
        engine_allie.quit()
        engine_enemy.quit()
        sys.exit()

    vis = visualizer.Visualizer()

    # clock = pygame.time.Clock()
    while True:
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    quit()
                if event.key == pygame.K_f:
                    is_white = not is_white
                    pygame.display.set_caption("White" if is_white else "Black")
                if event.key == pygame.K_c:
                    pyperclip.copy(board.fen())
                if event.key == pygame.K_v:
                    try:
                        board.set_fen(pyperclip.paste())
                    except ValueError:
                        print("invalid board")
                if event.key == pygame.K_r:
                    engine_allie_board = None
                    engine_enemy_board = None
                if event.key == pygame.K_s:
                    if engine_allie_result:
                        engine_allie_result.stop()
                    if engine_enemy_result:
                        engine_enemy_result.stop()
                if event.key == pygame.K_p:
                    image = ImageGrab.grab()
                    image = image.crop((0, 0, image.width//2+50, image.height))
                    is_match, tiles = vision.img2tiles(image)
                    if is_match:
                        board_temp = predictBoard(tiles, is_white)
                        encoded_filename = base64.b32encode(board_temp.fen().encode()).decode("ascii")
                        image.save(f"saved/{encoded_filename}.png")
                        print(f"image stored in saved/{encoded_filename}.png")
                    else:
                        print("no match found")

        image = ImageGrab.grab()
        image = image.crop((0, 0, image.width//2+50, image.height))
        is_match, tiles = vision.img2tiles(image)
        if is_match:
            board = predictBoard(tiles, is_white)

        board_temp = board.copy()
        board_temp.turn = chess.WHITE if is_white else chess.BLACK
        board_temp.castling_rights = board_temp.clean_castling_rights()

        if board_temp.is_valid() and board_temp != engine_allie_board:
            engine_allie_board = board_temp.copy()
            if engine_allie_result:
                engine_allie_result.stop()
            engine_allie_result = engine_allie.analysis(engine_allie_board, limit=chess.engine.Limit(depth=18), multipv=500)

        board_temp.turn = chess.BLACK if is_white else chess.WHITE
        board_temp.castling_rights = board_temp.clean_castling_rights()

        if board_temp.is_valid() and board_temp != engine_enemy_board:
            engine_enemy_board = board_temp.copy()
            if engine_enemy_result:
                engine_enemy_result.stop()
            engine_enemy_result = engine_enemy.analysis(engine_enemy_board, limit=chess.engine.Limit(depth=18), multipv=24)

        multipv_allie = engine_allie_result.multipv if engine_allie_result else [{}]
        multipv_enemy = engine_enemy_result.multipv if engine_enemy_result else [{}]
        vis.render_frame(engine_allie_board, multipv_allie, engine_enemy_board, multipv_enemy, is_white, board_temp)

        # print(clock.tick(3))

if __name__ == '__main__':
    main()
