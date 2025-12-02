const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureBtn = document.getElementById('capture-btn');
const resultCard = document.getElementById('result');
const personName = document.getElementById('person-name');
const confidenceScore = document.getElementById('confidence-score');
const loader = document.getElementById('loader');
const statusContent = document.getElementById('status-content');

// Access Webcam
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    } catch (err) {
        console.error("Error accessing webcam:", err);
        alert("Could not access webcam. Please allow permissions.");
    }
}

startCamera();

captureBtn.addEventListener('click', async () => {
    // UI Feedback
    captureBtn.disabled = true;
    resultCard.classList.remove('hidden');
    loader.classList.remove('hidden');
    statusContent.style.display = 'none';
    resultCard.className = 'result-card'; // Reset classes

    // Capture frame
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to blob
    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'capture.jpg');

        try {
            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            // Update UI
            loader.classList.add('hidden');
            statusContent.style.display = 'block';
            
            if (data.error) {
                personName.innerText = "Error";
                confidenceScore.innerText = data.error;
                resultCard.classList.add('access-denied');
            } else {
                personName.innerText = data.person;
                confidenceScore.innerText = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
                
                if (data.person !== "Unknown") {
                    resultCard.classList.add('access-granted');
                } else {
                    resultCard.classList.add('access-denied');
                }
            }
        } catch (error) {
            console.error(error);
            loader.classList.add('hidden');
            statusContent.style.display = 'block';
            personName.innerText = "Connection Error";
            confidenceScore.innerText = "Is the backend running?";
            resultCard.classList.add('access-denied');
        } finally {
            captureBtn.disabled = false;
        }
    }, 'image/jpeg');
});
