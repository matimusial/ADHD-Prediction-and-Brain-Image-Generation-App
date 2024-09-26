document.addEventListener('DOMContentLoaded', () => {
    const X_CLASS = 'x';
    const CIRCLE_CLASS = 'circle';
    const WINNING_COMBINATIONS = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6]
    ];

    const cellElements = document.querySelectorAll('[data-cell]');
    const board = document.getElementById('ticTacToeBoard');
    const winningMessageElement = document.getElementById('winningMessage');
    const restartButton = document.getElementById('restartButton');
    const winnerElement = document.getElementById('winner');
    let circleTurn;

    startGame();

    restartButton.addEventListener('click', startGame);

    function startGame() {
        circleTurn = true;
        cellElements.forEach(cell => {
            cell.classList.remove(X_CLASS, CIRCLE_CLASS);
            cell.innerHTML = '';
            cell.addEventListener('click', handleClick, { once: true });
        });
        setBoardHoverClass();
        winningMessageElement.classList.add('hidden');
    }

    function handleClick(e) {
        const cell = e.target;
        if (cell.classList.contains(X_CLASS) || cell.classList.contains(CIRCLE_CLASS)) return;

        const currentClass = circleTurn ? CIRCLE_CLASS : X_CLASS;
        placeMark(cell, currentClass);

        if (checkWin(currentClass)) {
            endGame(false);
        } else if (isDraw()) {
            endGame(true);
        } else {
            swapTurns();
            setBoardHoverClass();
            if (!circleTurn) makeBestMove();
        }
    }

    function endGame(draw) {
        winnerElement.innerText = draw ? 'Draw!' : `${circleTurn ? "Brain" : "Computer"} wins!`;
        winningMessageElement.classList.remove('hidden');
    }

    function isDraw() {
        return [...cellElements].every(cell => cell.classList.contains(X_CLASS) || cell.classList.contains(CIRCLE_CLASS));
    }

    function placeMark(cell, currentClass) {
        if (currentClass === CIRCLE_CLASS) {
            const img = document.createElement('img');
            img.src = '../resources/pilka.png';
            cell.appendChild(img);
        } else {
            cell.innerText = 'X';
        }
        cell.classList.add(currentClass);
    }

    function swapTurns() {
        circleTurn = !circleTurn;
    }

    function setBoardHoverClass() {
        board.classList.remove(X_CLASS, CIRCLE_CLASS);
        board.classList.add(circleTurn ? CIRCLE_CLASS : X_CLASS);
    }

    function checkWin(currentClass) {
        return WINNING_COMBINATIONS.some(combination => combination.every(index => cellElements[index].classList.contains(currentClass)));
    }

    function makeBestMove() {
        const move = Math.random() < 0.2 ? getRandomMove() : minimax([...cellElements], X_CLASS);
        const cell = cellElements[typeof move === 'number' ? move : move.index];
        placeMark(cell, X_CLASS);
        if (checkWin(X_CLASS)) endGame(false);
        else if (isDraw()) endGame(true);
        else {
            swapTurns();
            setBoardHoverClass();
        }
    }

    function getRandomMove() {
        const availableCells = [...cellElements].map((cell, index) => !cell.classList.contains(X_CLASS) && !cell.classList.contains(CIRCLE_CLASS) ? index : null).filter(index => index !== null);
        const randomIndex = Math.floor(Math.random() * availableCells.length);
        return availableCells[randomIndex];
    }

    function minimax(newBoard, player) {
        const availSpots = newBoard.filter(cell => !cell.classList.contains(X_CLASS) && !cell.classList.contains(CIRCLE_CLASS));

        if (checkWinAI(newBoard, CIRCLE_CLASS)) return { score: -10 };
        if (checkWinAI(newBoard, X_CLASS)) return { score: 10 };
        if (availSpots.length === 0) return { score: 0 };

        const moves = availSpots.map(spot => {
            const move = { index: newBoard.indexOf(spot) };
            newBoard[move.index].classList.add(player);
            move.score = player === X_CLASS ? minimax(newBoard, CIRCLE_CLASS).score : minimax(newBoard, X_CLASS).score;
            newBoard[move.index].classList.remove(player);
            return move;
        });

        return player === X_CLASS
            ? moves.reduce((bestMove, move) => move.score > bestMove.score ? move : bestMove, { score: -Infinity })
            : moves.reduce((bestMove, move) => move.score < bestMove.score ? move : bestMove, { score: Infinity });
    }

    function checkWinAI(board, player) {
        return WINNING_COMBINATIONS.some(combination => combination.every(index => board[index].classList.contains(player)));
    }
});
