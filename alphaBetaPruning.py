from types import NoneType
import chess
import keras
import dataProccesing
import numpy as np
import time

board = chess.Board()
evalModel = keras.models.load_model("chessModelEval.keras")
positionModel = keras.models.load_model("chessModel.keras")
intMove = np.load("numTypeMoves.npy", allow_pickle=True).item()
intMove = dict(zip(intMove.values(), intMove.keys()))

def sumPieces(board:chess.Board):
    posSum = 0
    pieceTotal = {chess.PAWN: 1, chess.BISHOP: 3.2, chess.KNIGHT: 3, chess.KING: 200, chess.QUEEN: 10, chess.ROOK: 5}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece != None:

            if piece.color == chess.WHITE:
                posSum += pieceTotal[piece.piece_type]
            else:
                posSum -= pieceTotal[piece.piece_type]
    posSum += 1 if board.has_castling_rights(chess.WHITE) else 0
    posSum -= 1 if board.has_castling_rights(chess.BLACK) else 0
    
    return posSum
def eval(board):
    matrixBoard = dataProccesing.boardToMatrix(board)
    matrixBoard = np.expand_dims(matrixBoard, axis=0)
    
    evalNN = evalModel.predict(matrixBoard, verbose=0)
    evalSum = sumPieces(board)
    total = evalNN+evalSum
    return total


def getBestMoves(board: chess.Board, numMoves):
    matrixBoard = dataProccesing.boardToMatrix(board)
    matrixBoard = np.expand_dims(matrixBoard, axis=0)
    moves = positionModel.predict(matrixBoard, verbose=0)
    bestMoves = np.argsort(moves[0])

    bestMoves = bestMoves[bestMoves.shape[0] - numMoves : bestMoves.shape[0]]
    legalBest40Moves = []
    allLegalMoves = list(board.generate_legal_moves())
    for i in bestMoves:
        if intMove[i] in allLegalMoves:
            legalBest40Moves.append(intMove[i])
    return legalBest40Moves


def get_user_move(board):
    legal_moves = [move.uci() for move in board.legal_moves]
    while True:
        user_move = input("Enter your move in UCI format (e.g., e2e4): ")
        if user_move in legal_moves:
            return chess.Move.from_uci(user_move)
        print("Invalid move. Please try again.")


def minMax(depth, color, alpha, beta, fen: str = chess.STARTING_BOARD_FEN):
    """
    This function is called by the make move function in order to evaluate a tree of all possible moves after devling into it by the given depth

    Args:
        fen(String): The FEN of the current Position
        depth(Integer): The depth that the function needs to recurse too
        alpha(Integer): The current highest value found in the tree, if the current value is greater, than we break out of the loop and return the
        beta(Integer): The current lowest value found in the tree, if the current value is less than alpha, we break out of the loop
        Maximising(Bool): Are we maximising for ourselves

    Returns:
        Float: Returns the best evaluation of the move
    """
    Maximising = True if color == "W" else False
    Tempboard = chess.Board(fen=fen)

    if Tempboard.is_game_over() or depth == 0:
        if not Tempboard.is_checkmate():
            return eval(board)
        if Tempboard.is_checkmate() and Maximising:
            return 1000000
        elif Tempboard.is_checkmate() and not Maximising:
            return -1000000

    if Maximising:
        curr_best = float("-inf")
        val_moves = list(getBestMoves(Tempboard, 30))
        for move in val_moves:
            Tempboard.push(move=move)
            if Tempboard.can_claim_draw():
                return -1000 if color == "W" else 10000
            val = minMax(
                fen=Tempboard.fen(), depth=depth - 1, alpha=alpha, beta=beta, color="B"
            )
            Tempboard.pop()

            val = max(val, curr_best)
            curr_best = val if max(val, curr_best) != curr_best else curr_best
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        return curr_best
    else:

        curr_best = float("inf")
        val_moves = list(getBestMoves(Tempboard, 30))

        for move in val_moves:
            Tempboard.push(move=move)
            val = minMax(
                fen=Tempboard.fen(), depth=depth - 1, alpha=alpha, beta=beta, color="W"
            )
            Tempboard.pop()
            if val < curr_best:
                curr_best = val
            beta = min(beta, val)

            if beta <= alpha:
                break
        return curr_best


def thing(currBoard):
    color = "B" if not currBoard.turn else "W"
    bestEval = float("-inf") if color == "W" else float("inf")
    bestMove = None
    startTime = time.process_time()
    currentDepth = 0
    if (currBoard.fullmove_number) > 2:
        currentDepth = 2
    elif (currBoard.fullmove_number) > 6:
        currBoard = ((currBoard.fullmove_number)) // 6
    for i in list(getBestMoves(currBoard, 20)):
        move = i
        currBoard.push(i)
        evalOfPos = minMax(
            currentDepth, color, float("-inf"), float("inf"), fen=currBoard.fen()
        )

        if color == "W":
            if evalOfPos > bestEval:
                bestMove = move
                bestEval = evalOfPos
        else:
            if evalOfPos < bestEval:
                bestMove = move
                bestEval = evalOfPos

        currBoard.pop()
    return bestMove


bestMove = None
while not board.is_game_over():
    thingMove = thing(board)
    print(thingMove)
    board.push(thingMove)
