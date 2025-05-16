window.onload = () => {
  const canvas = document.getElementById("drawingCanvas");
  const ctx = canvas.getContext("2d");

  // Yüksek çözünürlüklü beyaz zemin
  ctx.fillStyle = "#FFFFFF";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  let painting = false;
  const brushSize = document.getElementById("brushSize");
  const brushColor = document.getElementById("brushColor");

  function startPosition(e) {
    painting = true;
    draw(e);
  }
  function endPosition() {
    painting = false;
    ctx.beginPath();
  }
  function draw(e) {
    if (!painting) return;

    // Fare/touch koordinatını canvas koordinatına birebir eşit alıyoruz
    const rect = canvas.getBoundingClientRect();
    let x = (e.clientX || e.touches[0].clientX) - rect.left;
    let y = (e.clientY || e.touches[0].clientY) - rect.top;

    ctx.lineWidth = brushSize.value;
    ctx.lineCap = "round";
    ctx.strokeStyle = brushColor.value;
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
    e.preventDefault();
  }

  // Olay bağlamaları
  canvas.addEventListener("mousedown", startPosition);
  canvas.addEventListener("mouseup", endPosition);
  canvas.addEventListener("mousemove", draw);
  canvas.addEventListener("touchstart", startPosition, { passive: false });
  canvas.addEventListener("touchend", endPosition, { passive: false });
  canvas.addEventListener("touchmove", draw, { passive: false });

  // Temizle
  document.getElementById("clearButton").addEventListener("click", () => {
    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
  });

  // Kaydet (256×256 Quick Draw formatı)
  document.getElementById("saveButton").addEventListener("click", () => {
    const temp = document.createElement("canvas");
    const tctx = temp.getContext("2d");
    temp.width = 256;
    temp.height = 256;
    tctx.fillStyle = "#FFFFFF";
    tctx.fillRect(0, 0, 256, 256);
    tctx.drawImage(canvas, 0, 0, 256, 256);
    const data = temp.toDataURL("image/png");
    document.getElementById("imageData").value = data;
    document.getElementById("saveForm").submit();
  });
};
