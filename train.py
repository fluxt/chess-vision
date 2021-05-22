import base64
import glob
import re

import chess
import numpy as np
from joblib import dump, load
from PIL import Image, ImageGrab
from sklearn.neural_network import MLPClassifier

import vision

if __name__ == "__main__":
    images_dirs = sorted(glob.glob("images/*.png"))
    images = [Image.open(f) for f in images_dirs]
    images_tiles = []
    for image in images:
        is_match, tiles = vision.img2tiles(image)
        if is_match:
            images_tiles.append(tiles)
    X = np.array(images_tiles).reshape([-1, 1024])

    fens = [re.search(r"images/([a-zA-Z0-9=]*).png", f).group(1) for f in images_dirs]
    fens = [base64.b32decode(f).decode("ascii") for f in fens]
    boards = [chess.Board(f) for f in fens]
    pieces = [board.piece_at(square) for board in boards for square in range(64)]
    pieces = [piece.symbol() if piece != None else " " for piece in pieces]
    y = np.array(pieces)

    clf = MLPClassifier(max_iter=1000)
    clf.fit(X, y)
    print(clf.score(X, y))


    dump(clf, "models/clf.joblib")

