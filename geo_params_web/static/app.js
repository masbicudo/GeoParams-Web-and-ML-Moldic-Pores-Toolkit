function initializeHelpTooltips() {
    if (!window.bootstrap || !window.bootstrap.Tooltip) return;
    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(element => {
        window.bootstrap.Tooltip.getOrCreateInstance(element, {
            trigger: 'hover focus',
        });
    });
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeHelpTooltips);
} else {
    initializeHelpTooltips();
}
window.addEventListener('load', initializeHelpTooltips);

if (current_endpoint == "index")
{

}

if (current_endpoint == "image_select")
{
    const selectionInfo = document.getElementById("selectionInfo");

    function selectedImage() {
        const imageId = document.getElementById("image_id").value;
        return imagesAvailable.find(item => item.id === imageId);
    }

    function showSelectedImage() {
        const selected = selectedImage();
        if (!selected) return;
        document.getElementById('main-img').src = selected.preview_url;
        for (const field of ["x", "y", "w", "h"]) {
            document.getElementById(field).value = "0";
        }
        selectionBox.style.display = 'none';
        selectionInfo.style.display = 'none';
    }

    document.getElementById("image_id").addEventListener("change", showSelectedImage);

    document.addEventListener("DOMContentLoaded", function () {
        showSelectedImage();
    });

    const image = document.getElementById('main-img');
    const selectionBox = document.getElementById('selectionBox');
    let startX, startY;
    
    function moveInfoBox(x, y, w, h){
        const margin = 10;
        const infoBoxWidth = selectionInfo.offsetWidth;
        let infoLeft = x + w + margin;
        if (infoLeft + infoBoxWidth > image.clientWidth) {
            infoLeft = x - infoBoxWidth - margin;
        }
        selectionInfo.style.left = `${Math.max(0, infoLeft)}px`;
        selectionInfo.style.top = `${Math.max(0, y)}px`;
    }

    image.addEventListener('mousedown', (e) => {
        if (e.button != 0)
            return;

        e.preventDefault();

        const rect = image.getBoundingClientRect();
        const scaleX = image.naturalWidth / image.clientWidth;
        const scaleY = image.naturalHeight / image.clientHeight;
        startX = e.clientX - rect.left;
        startY = e.clientY - rect.top;

        selectionBox.style.left = startX + 'px';
        selectionBox.style.top = startY + 'px';
        selectionBox.style.width = '0px';
        selectionBox.style.height = '0px';
        selectionBox.style.display = 'block';
        selectionInfo.style.display = 'block';

        function onMouseMove(eMove) {

            const currentX = eMove.clientX - rect.left;
            const currentY = eMove.clientY - rect.top;

            const x = Math.min(startX, currentX);
            const y = Math.min(startY, currentY);
            const w = Math.abs(currentX - startX);
            const h = Math.abs(currentY - startY);

            area_px = (Math.round(w * h * scaleX * scaleY));
            if (area_px > 100000)
                return

            selectionBox.style.left = x + 'px';
            selectionBox.style.top = y + 'px';
            selectionBox.style.width = w + 'px';
            selectionBox.style.height = h + 'px';

            const widthPixels = Math.round(w * scaleX);
            const heightPixels = Math.round(h * scaleY);
            const selected = selectedImage();
            selectionInfo.innerHTML = `${widthPixels}px x ${heightPixels}px — ${area_px} px²`;
            if (selected && selected.mm_per_pixel) {
                const ratio = selected.mm_per_pixel;
                const px2mm = value => Math.round(value * ratio * 100)/100;
                const px2mmSq = value => Math.round(value * ratio * ratio * 100)/100;
                selectionInfo.innerHTML = `${px2mm(widthPixels)}mm x ${px2mm(heightPixels)}mm — ${px2mmSq(area_px)} mm²<br/>` + selectionInfo.innerHTML;
            }
            moveInfoBox(x, y, w, h);
        }

        function onMouseUp(eUp) {
            const boxRect = selectionBox.getBoundingClientRect();
            const imageRect = image.getBoundingClientRect();

            // Position relative to the image:
            const x = boxRect.left - imageRect.left;
            const y = boxRect.top - imageRect.top;
            const w = boxRect.width;
            const h = boxRect.height;


            // Convert to image pixel coordinates
            const imgX = Math.round(x * scaleX);
            const imgY = Math.round(y * scaleY);
            const imgW = Math.round(w * scaleX);
            const imgH = Math.round(h * scaleY);

            document.getElementById('x').value = imgX;
            document.getElementById('y').value = imgY;
            document.getElementById('w').value = imgW;
            document.getElementById('h').value = imgH;

            if (imgW == 0 || imgH == 0) {
                selectionBox.style.display = 'none';
                selectionInfo.style.display = 'none';
            }
            
            image.removeEventListener('mousemove', onMouseMove);
            image.removeEventListener('mouseup', onMouseUp);
        }

        image.addEventListener('mousemove', onMouseMove);
        image.addEventListener('mouseup', onMouseUp);
    });

    window.addEventListener('resize', () => {

        const scaleX = image.clientWidth / image.naturalWidth;
        const scaleY = image.clientHeight / image.naturalHeight;

        imgX = document.getElementById('x').value,
        imgY = document.getElementById('y').value,
        imgW = document.getElementById('w').value,
        imgH = document.getElementById('h').value

        if (imgW == 0 || imgH == 0)
            return;

        const x = imgX * scaleX;
        const y = imgY * scaleY;
        const w = imgW * scaleX;
        const h = imgH * scaleY;
    
        selectionBox.style.left = x + 'px';
        selectionBox.style.top = y + 'px';
        selectionBox.style.width = w + 'px';
        selectionBox.style.height = h + 'px';
        selectionBox.style.display = 'block';
        
        moveInfoBox(x, y, w, h);
    });
    
}

if (current_endpoint == "params_select")
{
    const params_img = document.getElementById('main-img');
    const preview_img = document.getElementById('preview-img');
    const slider = document.getElementById("my-slider");
    const sliderValueDisplay = document.getElementById("slider-value");

    clickedPoints.forEach(pt => {
        appendXYCoordsToListUI(pt.x, pt.y);
    });

    params_img.addEventListener('mousemove', function (e) {
        const rect = params_img.getBoundingClientRect();
        const scaleX = params_img.naturalWidth / params_img.width;
        const scaleY = params_img.naturalHeight / params_img.height;
        const x = Math.floor((e.clientX - rect.left) * scaleX);
        const y = Math.floor((e.clientY - rect.top) * scaleY);
    
        const bgX = -(x * tile_shape[1]);
        const bgY = -(y * tile_shape[0]);

        preview_img.style.backgroundPosition = `${bgX}px ${bgY}px`;

    });
    params_img.addEventListener('click', async function (event) {
        const img = event.target;
    
        // Get real size vs. displayed size
        const rect = img.getBoundingClientRect();
        const scaleX = img.naturalWidth / rect.width;
        const scaleY = img.naturalHeight / rect.height;
    
        // Mouse position relative to image
        const x = Math.floor((event.clientX - rect.left) * scaleX);
        const y = Math.floor((event.clientY - rect.top) * scaleY);
    
        try {
            const resp = await fetch(`/add_point?session_id=${sessionId}`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({x, y})
            });
    
            const result = await resp.json();
    
            if (resp.ok && result.status === 'ok') {
                appendXYCoordsToListUI(x, y);
            } else if (result.status === 'duplicate') {
                console.log("Point already exists:", result.point);
            } else {
                alert("Failed to add point.");
            }
        } catch (err) {
            console.error("Error adding point:", err);
            alert("An error occurred when adding the point.");
        }
    
    });
    function appendXYCoordsToListUI(x, y) {
        // Append to list
        const list = document.getElementById('coord-list');
        const item = document.createElement('li');
        item.classList.add('list-group-item')
        item.classList.add('col-6')
        item.classList.add('col-md-2')
        item.textContent = `x=${x}, y=${y}`;

        const removeBtn = document.createElement('span');
        removeBtn.textContent = ' ❌'; // space + ×
        removeBtn.classList.add('remove-btn');
        removeBtn.style.cursor = 'pointer';
        removeBtn.style.color = 'red';
        removeBtn.style.marginLeft = '10px';
        removeBtn.onclick = async function () {
            try {
                const deleteResp = await fetch(`/delete_point?session_id=${sessionId}`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({x, y})
                });
                const deleteResult = await deleteResp.json();
        
                if (deleteResp.ok && deleteResult.status === "ok") {

                    const { x: dx, y: dy } = deleteResult.deleted;

                    // Iterate and remove all matching <li> elements
                    const list = document.getElementById('coord-list');
                    const items = list.querySelectorAll('li');
                    items.forEach(li => {
                        const match = li.textContent.match(/x=(\d+), y=(\d+)/);
                        if (match) {
                            const lx = parseInt(match[1]);
                            const ly = parseInt(match[2]);
                            if (lx === dx && ly === dy) {
                                li.remove();
                            }
                        }
                    });
        
                } else {
                    alert(`Server could not delete point: (${x}, ${y})`);
                }
            } catch (err) {
                console.error("Error deleting point:", err);
                alert("An error occurred while trying to delete the point.");
            }
        
        };
        item.appendChild(removeBtn);
        list.appendChild(item);
    }

    function updateImages() {
        if (!tile_shape) return;
    
        const timestamp = new Date().getTime(); // or Math.random()
        params_img.src = `/static/output/${sessionId}/${counter}/main_image.png?t=${timestamp}`;

        const tileHeight = tile_shape[0];
        const tileWidth = tile_shape[1];
    
        const offsetX = 0;
        const offsetY = 0;
    
        preview_img.style.width = tileWidth + "px";
        preview_img.style.height = tileHeight + "px";
        preview_img.style.backgroundImage = `url(/static/output/${sessionId}/${counter}/stitched_tiles.png?t=${timestamp})`;
        preview_img.style.backgroundPosition = `${offsetX}px ${offsetY}px`;
        preview_img.style.backgroundRepeat = "no-repeat";
        preview_img.style.imageRendering = "pixelated";  // optional
    }
    
    function fetchTaskInfo(name) {
        fetch("/get_task_info/" + name + "?session_id=" + sessionId)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    console.log("No task yet.");
                    document.getElementById("progress").innerText =
                        "Error processing! Go back and try again.";
                    return;
                }
                var task = data.task;
                if (task.state == "Done") {
                    tile_shape = task.result.tile_shape
                    document.getElementById("progress").innerText =
                        lr.str.finished_processing;
                    updateImages();
                    document.getElementById("everything").classList.remove("hidden");
                    document.getElementById("hide-when-processing-done").classList.add("hidden");
                }
                if (task.state == "Requested") {
                    document.getElementById("everything").classList.remove("hidden");
                    document.getElementById("hide-when-processing-done").classList.add("hidden");
                    if (task.progress) {
                        const p = task.progress.step/task.progress.total_steps;
                        document.getElementById("progress").innerText =
                            lr.str.progress.replace("{p}", (p * 100).toFixed(0));
                    } else {
                        document.getElementById("progress").innerText =
                            lr.str.processing;
                    }
                    setTimeout(() => fetchTaskInfo(name), 1000);
                }
            });
    }
    
    document.addEventListener("DOMContentLoaded", function () {
        fetchTaskInfo("initial_image_setup");
        setSliderLabel();
    });


    function setSliderLabel() {
        if (mmPerPixel) {
            const mm2 = (slider.value * mmPerPixel * mmPerPixel).toFixed(4);
            sliderValueDisplay.textContent = slider.value + "px² = " + mm2 + "mm²";
        } else {
            sliderValueDisplay.textContent = slider.value + "px² (physical scale not provided)";
        }
    }
    slider.addEventListener("input", function () {
        setSliderLabel();
    });

    // Submit slider value to Flask
    function submitSlider() {
        fetch(`/params_select?session_id=${sessionId}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ min_pore_size: slider.value })
        })
        .then(response => response.json())
        .then(data => {
            fetchTaskInfo("initial_image_setup");
        });
    }

    async function restart_async(reason) {
        priority = document.getElementById("priority").value;
        response = await fetch(`/params_select?session_id=${sessionId}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ end_reason: reason, priority })
        });
        if (response.ok) {
            const data = await response.json();
            if (data.status !== "canceled" && data.status !== "done" && data.status !== "error") {
                alert("Failed to restart the process.");
                return;
            }
            if (data.status === "error") {
                window.location.reload();
                return;
            }
            if (data.status === "done") {
                window.location.href = "/end_review?session_id=" + sessionId;
                return;
            }
        } else {
            alert("Failed to restart the process.");
            return;
        }
        window.location.href = "/?session_id=" + sessionId + "&renew=True";
    }
    function restart(reason) {
        restart_async(reason);
    }
}

if (current_endpoint == "dataset_detail")
{
    document.querySelectorAll(".calibration-card").forEach(card => {
        const image = card.querySelector(".calibration-image");
        const overlay = card.querySelector(".calibration-overlay");
        const line = card.querySelector(".calibration-line");
        const pointOne = card.querySelector(".point-one");
        const pointTwo = card.querySelector(".point-two");
        const distanceInput = card.querySelector(".pixel-distance");
        const status = card.querySelector(".calibration-status");
        let points = [];

        function draw() {
            const width = image.clientWidth;
            const height = image.clientHeight;
            overlay.setAttribute("viewBox", `0 0 ${width} ${height}`);
            const elements = [pointOne, pointTwo];
            elements.forEach((point, index) => {
                const value = points[index];
                point.style.display = value ? "block" : "none";
                if (value) {
                    point.setAttribute("cx", value.x);
                    point.setAttribute("cy", value.y);
                }
            });
            if (points.length === 2) {
                line.style.display = "block";
                line.setAttribute("x1", points[0].x);
                line.setAttribute("y1", points[0].y);
                line.setAttribute("x2", points[1].x);
                line.setAttribute("y2", points[1].y);
            } else {
                line.style.display = "none";
            }
        }

        image.addEventListener("click", event => {
            const rect = image.getBoundingClientRect();
            const point = {
                x: event.clientX - rect.left,
                y: event.clientY - rect.top,
            };
            if (points.length === 2) points = [];
            points.push(point);
            if (points.length === 2) {
                const displayedDistance = Math.hypot(
                    points[1].x - points[0].x,
                    points[1].y - points[0].y,
                );
                const previewDistance = displayedDistance * image.naturalWidth / image.clientWidth;
                distanceInput.value = previewDistance.toFixed(4);
                status.textContent = `Marked scale bar: ${previewDistance.toFixed(1)} preview pixels. Enter its physical length and save.`;
            } else {
                distanceInput.value = "";
                status.textContent = "Click the other endpoint of the scale bar.";
            }
            draw();
        });
        window.addEventListener("resize", () => {
            points = [];
            distanceInput.value = "";
            draw();
        });
        draw();
    });
}
