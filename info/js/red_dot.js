document.addEventListener('DOMContentLoaded', () => {
    const gameArea = document.getElementById('gameArea');
    const target = document.getElementById('target');
    const scoreDisplay = document.getElementById('score');
    const averageTimeDisplay = document.getElementById('averageTime');

    let score = 0;
    const clickTimes = [];
    let lastClickTime = null;
    let gameStarted = false;

    function moveTarget() {
        const { width: maxX, height: maxY } = gameArea.getBoundingClientRect();
        const randomX = Math.floor(Math.random() * (maxX - target.clientWidth));
        const randomY = Math.floor(Math.random() * (maxY - target.clientHeight));

        target.style.left = `${randomX}px`;
        target.style.top = `${randomY}px`;
    }

    function calculateAverageTime(times) {
        const sum = times.reduce((a, b) => a + b, 0);
        return sum / times.length;
    }

    target.addEventListener('click', () => {
        const currentTime = Date.now();

        if (gameStarted) {
            const timeDifference = currentTime - lastClickTime;
            clickTimes.push(timeDifference);
        } else {
            gameStarted = true;
        }
        lastClickTime = currentTime;

        score++;
        scoreDisplay.textContent = score;

        if (score === 21) {
            const averageTime = calculateAverageTime(clickTimes);
            alert(`Avg time: ${averageTime.toFixed(2)} ms`);
            resetGame();
        }

        moveTarget();
    });

    function resetGame() {
        score = 0;
        scoreDisplay.textContent = score;
        clickTimes.length = 0;
        gameStarted = false;
        averageTimeDisplay.textContent = '';
    }

    moveTarget();
});
