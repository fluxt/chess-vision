# Live Chess Anaylzer

In online chess websites, when spectating bullet chess, such as 1+0 time control, it is often hard to guage the engine analysis because they tend to be in algebraic notation. By taking a screenshot of the image, extracting the FEN through TensorFlow, and visualizing the engine analysis through various opacities, we can predict potentially playable moves.

![](https://github.com/fluxt/chess-vision/blob/master/figures/demo.png)

## Credits

There is already many previous literature on chess image vision, including implementations. However there are very few implementations that run real-time. I have merely accelerated it slightly to get the application to run around 12fps.

 - [ChessVision](https://github.com/pmauchle/ChessVision)

 - [tensorflow_chessbot](https://github.com/Elucidation/tensorflow_chessbot)
