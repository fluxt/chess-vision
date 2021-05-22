import io
import os

import chess
import chess.engine
import chess.svg
import numpy as np
import pygame
import pyvips
from PIL import Image
from scipy.special import expit, logit

size = width, height = 900, 900
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)
dark_green = (0, 127, 0)
blue = (0, 0, 255)
black = (0, 0, 0)
font_dir = "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf"
font_bold_dir = "/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf"

def rotatesquare(square, is_white):
    return square if is_white else 63-square

def score2num(score):
    if score == chess.engine.MateGiven:
        return 9999
    if score.is_mate() and score.mate() > 0:
        return 9999 - score.mate() * 100
    if score.is_mate() and score.mate() <= 0:
        return -9999 - score.mate() * 100
    return score.score()

def calc_pl(eval, board):
    # lambda(eval), lambda(-5)=2.6, lambda(0) = 0.5, lambda(5)=3.6
    weakness = 0
    if eval >= 0:
        weakness = 0.5+0.60*np.clip(np.abs(eval), 0.0, 7.5)
    else:
        weakness = 0.5+0.40*np.clip(np.abs(eval), 0.0, 7.5)
    num_pieces = len(board.piece_map())
    weakness = weakness * (1.0-0.3*expit((num_pieces-8)/2)) * 2.2
    return weakness

def svg2surface(svg):
    format_to_dtype = {
        'uchar': np.uint8,
        'char': np.int8,
        'ushort': np.uint16,
        'short': np.int16,
        'uint': np.uint32,
        'int': np.int32,
        'float': np.float32,
        'double': np.float64,
        'complex': np.complex64,
        'dpcomplex': np.complex128,
    }

    img_pv = pyvips.Image.new_from_buffer(svg.encode(), "")
    a = np.ndarray(buffer=img_pv.write_to_memory(),
                   dtype=format_to_dtype[img_pv.format],
                   shape=[img_pv.height, img_pv.width, img_pv.bands])
    png = Image.fromarray(a).resize((400, 400))
    return pygame.image.fromstring(png.tobytes(), png.size, png.mode).convert()

class Visualizer():
    def __init__(self):
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{1920//2+80}, {1080//2}"
        pygame.init()
        pygame.display.set_icon(pygame.image.fromstring(b'\x00', (1, 1), 'P'))
        self.screen = pygame.display.set_mode(size)

    def render_frame(self, board_allie, multipv_allie,
                           board_enemy, multipv_enemy,
                           is_white, board_screen):

        is_focused = pygame.mouse.get_focused()
        mousepos = pygame.mouse.get_pos()
        self.screen.fill(black)

        # draw board and get engine for allie's turn - IDENTICAL CODE
        board = board_allie
        multipv = multipv_allie

        arrows=[]
        if is_focused and 80 <= mousepos[0] <= 400 and 420 <= mousepos[1] <= 900:
            pvrank = (mousepos[1] - 420) // 20
            moveno = (mousepos[0] - 80) // 80
            if pvrank < len(multipv):
                pv = multipv[pvrank].get("pv")
                if pv:
                    for move in pv[:moveno+1]:
                        arrows.append((move.from_square, move.to_square))
        elif is_focused and 0 <= mousepos[0] <= 400 and 0 <= mousepos[1] <= 400:
            pass
        else:
            scores = [info.get("score") for info in multipv]
            scores = np.array([score2num(s.relative) if s else -9999 for s in scores]) / 100
            eval = scores[0]
            scores_diff = scores - eval
            scores_diff = np.exp(scores_diff / calc_pl(eval, board)) * 0.8
            scores_diff = np.clip(scores_diff, 0.0, 0.8)
            for info, opacity in zip(multipv[:24], scores_diff[:24]):
                score = info.get("score")
                pv = info.get("pv")
                if score and pv and opacity > 0.05:
                    arrows.append(chess.svg.Arrow(pv[0].from_square, pv[0].to_square, color=f"rgba({dark_green[0]},{dark_green[1]},{dark_green[2]},{opacity:.3f})"))

        board_svg = chess.svg.board(board=board_screen, size=400, orientation=chess.WHITE if is_white else chess.BLACK,
                                    coordinates=False, arrows=arrows)
        board_image_pygame = svg2surface(board_svg)
        self.screen.blit(board_image_pygame, (0,0))

        if is_focused and 0 <= mousepos[0] <= 400 and 0 <= mousepos[1] <= 400:
            rank = (mousepos[1] - 0) // 50
            file = (mousepos[0] - 0) // 50
            square = chess.square_mirror(chess.square(file, rank))
            square = rotatesquare(square, is_white)
            for i, info in enumerate(multipv):
                score = info.get("score")
                pv = info.get("pv")
                if score and pv:
                    score = score2num(score.relative)
                    move = pv[0]
                    if square == move.from_square:
                        to_square_draw = rotatesquare(move.to_square, is_white)
                        to_square_draw = chess.square_mirror(to_square_draw)
                        text = f"{score/100:.2f}"
                        text_pygame = pygame.font.Font(font_bold_dir, 16).render(text, True, dark_green)
                        text_pygame_rect = text_pygame.get_rect()
                        text_pygame_rect.center = (25+chess.square_file(to_square_draw)*50, 25+chess.square_rank(to_square_draw)*50)
                        self.screen.blit(text_pygame, text_pygame_rect)
                        pygame.draw.rect(self.screen, dark_green, (80, 420+i*20, 320, 20))

        text = ""

        if board.status() == chess.Status.VALID:
            depth = multipv[0].get("depth")
            nodes = multipv[0].get("nodes")
            text += f"breadth: {len(multipv)} depth: {depth} nodes: {nodes}\n"
        else:
            text += str(board.status()) + "\n"

        for info in multipv[:24]:
            if info.get("score"):
                score = score2num(info.get("score").relative)
                text += f"|{score/100:6.2f}|"
            board_temp = board.copy()
            for i, move in enumerate(info.get("pv")[:4]) if info.get("pv") else enumerate([]):
                text += f"|{board_temp.san(move):6s}|"
                board_temp.push(move)
            text += "\n"

        for i, line in enumerate(text.split("\n")):
            text_pygame = pygame.font.Font(font_dir, 20).render(line, True, white)
            self.screen.blit(text_pygame, (0,400+20*i))



        # draw board and get engine for enemy's turn - IDENTICAL CODE
        board = board_enemy
        multipv = multipv_enemy

        arrows=[]
        if is_focused and 580 <= mousepos[0] <= 900 and 420 <= mousepos[1] <= 900:
            pvrank = (mousepos[1] - 420) // 20
            moveno = (mousepos[0] - 580) // 80
            if pvrank < len(multipv):
                pv = multipv[pvrank].get("pv")
                if pv:
                    for move in pv[:moveno+1]:
                        arrows.append((move.from_square, move.to_square))
        elif is_focused and 500 <= mousepos[0] <= 900 and 0 <= mousepos[1] <= 400:
            pass
        else:
            scores = [info.get("score") for info in multipv]
            scores = np.array([score2num(s.relative) if s else -9999 for s in scores]) / 100
            eval = scores[0]
            scores_diff = scores - eval
            scores_diff = np.exp(scores_diff * calc_pl(eval, board)) * 0.8
            scores_diff = np.clip(scores_diff, 0.0, 0.8)
            for info, opacity in zip(multipv[:24], scores_diff[:24]):
                score = info.get("score")
                pv = info.get("pv")
                if score and pv and opacity > 0.05:
                    arrows.append(chess.svg.Arrow(pv[0].from_square, pv[0].to_square, color=f"rgba({dark_green[0]},{dark_green[1]},{dark_green[2]},{opacity:.3f})"))

        board_svg = chess.svg.board(board=board_screen, size=400, orientation=chess.WHITE if is_white else chess.BLACK,
                                    coordinates=False, arrows=arrows)
        board_image_pygame = svg2surface(board_svg)
        self.screen.blit(board_image_pygame, (500,0))

        if is_focused and 500 <= mousepos[0] <= 900 and 0 <= mousepos[1] <= 400:
            rank = (mousepos[1] - 0) // 50
            file = (mousepos[0] - 500) // 50
            square = chess.square_mirror(chess.square(file, rank))
            square = rotatesquare(square, is_white)
            for i, info in enumerate(multipv):
                score = info.get("score")
                pv = info.get("pv")
                if score and pv:
                    score = score2num(score.relative)
                    move = pv[0]
                    if square == move.from_square:
                        to_square_draw = rotatesquare(move.to_square, is_white)
                        to_square_draw = chess.square_mirror(to_square_draw)
                        text = f"{score/100:.2f}"
                        text_pygame = pygame.font.Font(font_bold_dir, 16).render(text, True, dark_green)
                        text_pygame_rect = text_pygame.get_rect()
                        text_pygame_rect.center = (525+chess.square_file(to_square_draw)*50, 25+chess.square_rank(to_square_draw)*50)
                        self.screen.blit(text_pygame, text_pygame_rect)
                        pygame.draw.rect(self.screen, dark_green, (580, 420+i*20, 320, 20))

        text = ""

        if board.status() == chess.Status.VALID:
            text += f"breadth: {len(multipv)} depth: {depth} nodes: {nodes}\n"
        else:
            depth = multipv[0].get("depth")
            nodes = multipv[0].get("nodes")
            text += str(board.status()) + "\n"

        for info in multipv[:24]:
            if info.get("score"):
                score = score2num(info.get("score").relative)
                text += f"|{score/100:6.2f}|"
            board_temp = board.copy()
            for i, move in enumerate(info.get("pv")[:4]) if info.get("pv") else enumerate([]):
                text += f"|{board_temp.san(move):6s}|"
                board_temp.push(move)
            text += "\n"

        for i, line in enumerate(text.split("\n")):
            text_pygame = pygame.font.Font(font_dir, 20).render(line, True, white)
            self.screen.blit(text_pygame, (500,400+20*i))



        pygame.display.flip()

