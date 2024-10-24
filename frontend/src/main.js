import { FaceDetection } from "@mediapipe/face_detection";
import { Camera } from "@mediapipe/camera_utils";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";

// إعداد Mediapipe لكشف الوجه
const progressCircle = document.getElementById("progressCircle");
const userInfoForm = document.getElementById("userInfoForm");
const videoElement = document.getElementById("videoInput");
const canvasElement = document.getElementById("outputCanvas");
const canvasCtx = canvasElement.getContext("2d");
const totalImages = 5;
const captureInterval = 1000; // الفاصل الزمني بين الالتقاطات بالمللي ثانية
let stopped = true;
let lastCaptureTime = 0; // وقت آخر التقاط
let capturedFaces = []; // مصفوفة لتخزين الصور الملتقطة

const faceDetection = new FaceDetection({
  locateFile: (file) => {
    return `/models/${file}`; // تحميل النماذج من CDN
  },
});

faceDetection.setOptions({
  model: "short",
  minDetectionConfidence: 0.5,
});

// كاميرا Mediapipe
const camera = new Camera(videoElement, {
  onFrame: async () => {
    if(!stopped)
      await faceDetection.send({ image: videoElement });
  },
  width: 900,
  height: 900,
});
camera.start();

faceDetection.onResults((results) => {
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  if (results.detections.length === 1) {
    const currentTime = Date.now(); // الحصول على الوقت الحالي
    const detection = results.detections[0];
    if (detection.boundingBox) {
      const box = detection.boundingBox;

      const x =
        box.xCenter * canvasElement.width -
        (box.width * canvasElement.width) / 2;
      const y =
        box.yCenter * canvasElement.height -
        (box.height * canvasElement.height) / 2;
      const width = box.width * canvasElement.width;
      const height = box.height * canvasElement.height;

      canvasCtx.strokeStyle = "green";
      canvasCtx.lineWidth = 2;
      canvasCtx.strokeRect(x, y, width, height);
      if (capturedFaces.length < totalImages) {
        // تحقق من الوقت بين الالتقاطات
        if (currentTime - lastCaptureTime >= captureInterval) {
          lastCaptureTime = currentTime;
          capturedFace(box); // التقاط الوجه مع الإحداثيات

          updateProgress((capturedFaces.length / totalImages) * 100);
        }
      } else {
        displayCapturedImages(capturedFaces);
      }
    }
  }
});

// التقاط صورة الوجه وقصها
async function capturedFace(box) {
  // توسيع حدود الوجه بنسبة 10%
  const scaleFactor = 1.4; // نسبة التوسيع
  let faceWidth = box.width * videoElement.videoWidth * scaleFactor;
  let faceHeight = box.height * videoElement.videoHeight * scaleFactor;
  let faceX = (box.xCenter * videoElement.videoWidth) - (faceWidth / 2);
  let faceY = (box.yCenter * videoElement.videoHeight) - (faceHeight / 2);

  // تأكد من عدم تجاوز حدود الفيديو
  faceX = Math.max(0, faceX);
  faceY = Math.max(0, faceY);
  faceWidth = Math.min(faceWidth, videoElement.videoWidth - faceX);
  faceHeight = Math.min(faceHeight, videoElement.videoHeight - faceY);

  // حفظ الوجه في مصفوفة كصورة
  const faceCanvas = document.createElement("canvas");
  faceCanvas.width = faceWidth;
  faceCanvas.height = faceHeight;
  const faceCtx = faceCanvas.getContext("2d");
  faceCtx.drawImage(
    videoElement,
    faceX,
    faceY,
    faceWidth,
    faceHeight,
    0,
    0,
    faceWidth,
    faceHeight
  );
  const faceImage = faceCanvas.toDataURL("image/jpeg");
  capturedFaces.push(faceImage); // إضافة الصورة إلى المصفوفة
}

userInfoForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  await sendFaceToBackend(capturedFaces);
  capturedFaces = []; // إعادة تعيين عدد الوجوه المكتشفة
});

function updateProgress(progress) {
  const percentage = Math.min(progress, 100); // ضمان عدم تجاوز 100%
  const angle = (percentage / 100) * 360; // تحويل النسبة المئوية إلى زاوية

  // تحديث نمط الدائرة
  progressCircle.style.background = `conic-gradient(#76c7c0 ${angle}deg, #e0e0e0 ${angle}deg)`;
}

// دالة لاستعراض الصور الملتقطة
function displayCapturedImages(images) {
  const container = document.getElementById("capturedImagesContainer");
  container.innerHTML = ""; // مسح الصور السابقة

  images.forEach((imageData) => {
    const img = document.createElement("img");
    img.src = imageData; // تعيين مصدر الصورة
    container.appendChild(img); // إضافة الصورة إلى الحاوية
  });
}

async function sendFaceToBackend(capturedFaces) {
  const response = await fetch("http://127.0.0.1:5000/api/extract-features", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      name: document.getElementById("username").value,
      userId: document.getElementById("userId").value,
      images: capturedFaces,
    }),
  });

  const data = await response.json();
  console.log("Extracted Features:", data);
}

document.getElementById("reset").onclick = () => {
  console.log('reset')
  capturedFaces = []; // مصفوفة لتخزين الصور الملتقطة
  updateProgress(0)
  displayCapturedImages([])
};

document.getElementById("start").onclick = () => {
  stopped = false
};

document.getElementById("stop").onclick = () => {
  stopped = true
};