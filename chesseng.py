import chess
import chess.engine

board = chess.Board()

engine = chess.engine.SimpleEngine.popen_uci(
    "/opt/homebrew/bin/stockfish"
)

def engine_reply(player_move):
    """
    player_move: e.g. 'e2e4'

    Returns:
        engine move as string, e.g. 'c7c5'
    """

    board.push_uci(player_move)

    result = engine.play(
        board,
        chess.engine.Limit(time=0.1)
    )

    engine_move = result.move

    board.push(engine_move)

    return engine_move.uci()

reply = engine_reply("e2e4")

print(reply)