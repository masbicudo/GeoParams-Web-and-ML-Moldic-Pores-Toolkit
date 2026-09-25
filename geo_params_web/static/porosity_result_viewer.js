(() => {
  const colors = {
    red: [255, 0, 0],
    black: [0, 0, 0],
    white: [255, 255, 255],
  };

  function loadImage(url) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error("The saved analysis image could not be loaded."));
      image.src = url;
    });
  }

  function setupViewer(viewer) {
    const canvas = viewer.querySelector("[data-porosity-canvas]");
    const maskRange = viewer.querySelector("[data-porosity-range]");
    const threshold = viewer.querySelector("[data-porosity-threshold]");
    const thresholdGroup = viewer.querySelector("[data-porosity-threshold-group]");
    const mode = viewer.querySelector("[data-porosity-mode]");
    const color = viewer.querySelector("[data-porosity-color]");
    const colorGroup = viewer.querySelector("[data-porosity-color-group]");
    const opacity = viewer.querySelector("[data-porosity-opacity]");
    const opacityOutput = viewer.querySelector("[data-porosity-opacity-output]");
    const selection = viewer.querySelector("[data-porosity-selection]");
    const error = viewer.querySelector("[data-porosity-viewer-error]");
    const zoomToggle = viewer.querySelector("[data-porosity-zoom]");
    const zoomLens = viewer.querySelector("[data-porosity-zoom-lens]");
    const zoomCanvas = viewer.querySelector("[data-porosity-zoom-canvas]");
    const zoomContext = zoomCanvas.getContext("2d");
    const normalizer = Number.parseFloat(viewer.dataset.normalizer) || 0;
    const context = canvas.getContext("2d");

    Promise.all([
      loadImage(viewer.dataset.inputUrl),
      loadImage(viewer.dataset.maskUrl),
    ]).then(([inputImage, maskImage]) => {
      const width = inputImage.naturalWidth;
      const height = inputImage.naturalHeight;
      canvas.width = width;
      canvas.height = height;

      const maskCanvas = document.createElement("canvas");
      maskCanvas.width = width;
      maskCanvas.height = height;
      const maskContext = maskCanvas.getContext("2d", { willReadFrequently: true });
      maskContext.imageSmoothingEnabled = false;
      maskContext.drawImage(maskImage, 0, 0, width, height);
      const maskPixels = maskContext.getImageData(0, 0, width, height).data;

      const renderedMask = document.createElement("canvas");
      renderedMask.width = width;
      renderedMask.height = height;
      const renderedContext = renderedMask.getContext("2d");
      let zoomPinned = false;
      let pinnedZoomPoint = null;

      function hideZoom() {
        zoomLens.classList.add("d-none");
      }

      function zoomPointFromEvent(event) {
        const bounds = canvas.getBoundingClientRect();
        const displayX = event.clientX - bounds.left;
        const displayY = event.clientY - bounds.top;
        return {
          pixelX: displayX * (width / bounds.width),
          pixelY: displayY * (height / bounds.height),
        };
      }

      function drawZoom(point) {
        const bounds = canvas.getBoundingClientRect();
        const displayX = point.pixelX * (bounds.width / width);
        const displayY = point.pixelY * (bounds.height / height);
        const sourceSize = Math.min(zoomCanvas.width, zoomCanvas.height) / 5;
        const sourceX = Math.max(
          0,
          Math.min(width - sourceSize, point.pixelX - sourceSize / 2),
        );
        const sourceY = Math.max(
          0,
          Math.min(height - sourceSize, point.pixelY - sourceSize / 2),
        );

        zoomContext.clearRect(0, 0, zoomCanvas.width, zoomCanvas.height);
        zoomContext.imageSmoothingEnabled = false;
        zoomContext.drawImage(
          canvas,
          sourceX,
          sourceY,
          sourceSize,
          sourceSize,
          0,
          0,
          zoomCanvas.width,
          zoomCanvas.height,
        );
        zoomLens.style.left = `${displayX}px`;
        zoomLens.style.top = `${displayY}px`;
        zoomLens.classList.remove("d-none");
      }

      function showZoom(event) {
        if (!zoomToggle.checked) {
          hideZoom();
          return;
        }
        if (!zoomPinned) {
          drawZoom(zoomPointFromEvent(event));
        }
      }

      function render() {
        const selected = threshold.selectedOptions[0];
        const selectedThreshold = Number.parseFloat(threshold.value);
        const selectedPorosity = Number.parseFloat(selected.dataset.porosity);
        const transparent = mode.value === "transparent";
        const binary = maskRange.value === "binary";
        const selectedColor = colors[color.value];
        const output = renderedContext.createImageData(width, height);
        const cutoff = selectedThreshold * normalizer;

        for (let index = 0; index < maskPixels.length; index += 4) {
          const gray = maskPixels[index];
          const pore = normalizer > 0 && gray >= cutoff;
          if (transparent) {
            output.data[index] = selectedColor[0];
            output.data[index + 1] = selectedColor[1];
            output.data[index + 2] = selectedColor[2];
            output.data[index + 3] = binary ? (pore ? 255 : 0) : gray;
          } else {
            const value = binary ? (pore ? 255 : 0) : gray;
            output.data[index] = value;
            output.data[index + 1] = value;
            output.data[index + 2] = value;
            output.data[index + 3] = 255;
          }
        }

        renderedContext.putImageData(output, 0, 0);
        context.clearRect(0, 0, width, height);
        context.globalAlpha = 1;
        context.drawImage(inputImage, 0, 0, width, height);
        context.globalAlpha = Number(opacity.value) / 100;
        context.drawImage(renderedMask, 0, 0);
        context.globalAlpha = 1;

        if (zoomPinned && pinnedZoomPoint) {
          drawZoom(pinnedZoomPoint);
        }

        thresholdGroup.classList.toggle("d-none", !binary);
        colorGroup.classList.toggle("d-none", !transparent);
        opacityOutput.value = `${opacity.value}%`;
        opacityOutput.textContent = `${opacity.value}%`;
        selection.textContent = `${selected.textContent.trim()} selected; ` +
          `preview opacity ${opacity.value}%.`;
      }

      [maskRange, threshold, mode, color, opacity].forEach((control) => {
        control.addEventListener("input", render);
        control.addEventListener("change", render);
      });
      zoomToggle.addEventListener("change", () => {
        zoomPinned = false;
        pinnedZoomPoint = null;
        if (!zoomToggle.checked) {
          hideZoom();
        }
      });
      canvas.addEventListener("pointermove", showZoom);
      canvas.addEventListener("pointerleave", () => {
        if (!zoomPinned) {
          hideZoom();
        }
      });
      canvas.addEventListener("click", (event) => {
        if (!zoomToggle.checked) {
          return;
        }
        if (zoomPinned) {
          zoomPinned = false;
          pinnedZoomPoint = null;
          hideZoom();
          return;
        }
        pinnedZoomPoint = zoomPointFromEvent(event);
        drawZoom(pinnedZoomPoint);
        zoomPinned = true;
      });
      render();
    }).catch((loadError) => {
      error.textContent = loadError.message;
      error.classList.remove("d-none");
      canvas.classList.add("d-none");
    });
  }

  document.querySelectorAll("[data-porosity-viewer]").forEach(setupViewer);
})();
