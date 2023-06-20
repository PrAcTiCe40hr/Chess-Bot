import chess
import torch
from chess_net import ChessNet

net = ChessNet()
net.load_state_dict(torch.load('chess_net.pth'))


def board_to_tensor(board):
    # Convert the board state to a PyTorch Tensor
    pass  # Implement this function
# This is our simple evaluation function.
# It simply counts the difference in number of pieces.


def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_checkmate():
        return evaluate(board)

    if maximizing_player:
        max_eval = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


# This function is used to find the best move for a given board state.
def find_best_move(board, depth):
    max_eval = float('-inf')
    best_move = None
    for move in board.legal_moves:
        board.push(move)
        eval = minimax(board, depth, float('-inf'), float('inf'), False)
        board.pop()
        if eval > max_eval:
            max_eval = eval
            best_move = move
    return best_move


def evaluate(board):
    # Convert the board state to a PyTorch Tensor
    # This will depend on how you chose to represent the board state
    board_state = board_to_tensor(board)

    # Pass the board state through the neural network
    output = net(board_state)

    # We use the sigmoid function to get a value between 0 and 1
    result = torch.sigmoid(output).item()

    # We want the evaluation to be between -1 and 1
    evaluation = 2*result - 1

    # If it's black's turn, we negate the evaluation
    if not board.turn:
        evaluation = - evaluation

    return evaluation


def main():
    board = chess.Board()
    while not board.is_checkmate():
        print(board)
        if board.turn:
            move = find_best_move(board, 2)
            if move is None:
                break
            board.push(move)
        else:
            print("Enter your move: ")
            move = input()
            board.push_san(move)
    print(board)


if __name__ == "__main__":
    main()
