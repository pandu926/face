const video = document.getElementById("video");

Promise.all([
  faceapi.nets.ssdMobilenetv1.loadFromUri("./models/models"), // Memuat model SsdMobilenetv1
  faceapi.nets.faceLandmark68Net.loadFromUri("./models"),
  faceapi.nets.faceRecognitionNet.loadFromUri("./models"),
  faceapi.nets.faceExpressionNet.loadFromUri("./models"),
])
  .then(startVideo)
  .catch((err) => console.log("Error loading models:", err));

function startVideo() {
  navigator.mediaDevices
    .getUserMedia({ video: {} })
    .then((stream) => (video.srcObject = stream))
    .catch((err) => console.error("Error accessing webcam:", err));
}

// Fungsi untuk memuat gambar-gambar wajah yang sudah dikenal
async function loadLabeledImages() {
  const labels = ["pandu"]; // Nama orang-orang yang akan dikenali
  return Promise.all(
    labels.map(async (label) => {
      const descriptions = [];
      for (let i = 1; i <= 3; i++) {
        // Asumsikan ada 3 gambar untuk setiap orang
        const img = await faceapi.fetchImage(
          `./models/wajah/${label}/${i}.jpeg`
        );
        const detections = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();
        if (detections && detections.descriptor) {
          descriptions.push(detections.descriptor);
          console.log(
            `Descriptor for ${label} image ${i}:`,
            detections.descriptor
          );
        } else {
          console.log(
            `No face detected or descriptor not found in ${label}/${i}.jpeg`
          );
        }
      }
      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
  );
}

video.addEventListener("play", async () => {
  const labeledFaceDescriptors = await loadLabeledImages();
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);

  const canvas = faceapi.createCanvasFromMedia(video);
  document.body.append(canvas);
  const displaySize = { width: video.width, height: video.height };
  faceapi.matchDimensions(canvas, displaySize);

  setInterval(async () => {
    try {
      const detections = await faceapi
        .detectAllFaces(video, new faceapi.SsdMobilenetv1Options()) // Menggunakan SsdMobilenetv1Options
        .withFaceLandmarks()
        .withFaceExpressions()
        .withFaceDescriptors();

      const resizedDetections = faceapi.resizeResults(detections, displaySize);

      canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);

      faceapi.draw.drawDetections(canvas, resizedDetections);
      faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
      faceapi.draw.drawFaceExpressions(canvas, resizedDetections);

      const results = resizedDetections.map((d) => {
        if (d.descriptor) {
          console.log("Face descriptor:", d.descriptor);
          return faceMatcher.findBestMatch(d.descriptor);
        } else {
          console.log("Face descriptor not found.");
          return { label: "Unknown", distance: 1 };
        }
      });

      const context = canvas.getContext("2d");

      results.forEach((result, i) => {
        const box = resizedDetections[i].detection.box;
        const text = result.toString();
        const x = box.x;
        const y = box.y - 10;

        context.fillStyle = "#00ff00";
        context.font = "16px Arial";
        context.fillText(text, x, y);

        // Redirect to dashboard if label is "pandu"
        // if (result.label === "pandu") {
        //   window.location.href = "/dashboard.html"; // Redirect ke halaman dashboard
        // }
      });

      // Jika tidak ada hasil yang cocok, tampilkan pesan error di kanvas
      const allUnknown = results.every((result) => result.label === "Unknown");
      if (allUnknown) {
        context.fillStyle = "#ff0000";
        context.font = "24px Arial";
        context.fillText("Error: Face not recognized", 10, 30);
      }
    } catch (error) {
      console.error("Error during face detection:", error);
    }
  }, 100);
});
