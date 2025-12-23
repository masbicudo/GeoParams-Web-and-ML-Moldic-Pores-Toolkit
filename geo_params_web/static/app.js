
if (current_endpoint == "index")
{

}

if (current_endpoint == "image_select")
{
    const selectionInfo = document.getElementById("selectionInfo");

    function fetchImageSz(name, percentage) {
        fetch("/make_thin_section?name=" + name + "&percentage=" + percentage)
            .then(response => response.json())
            .then(data => {
                const img = document.getElementById('main-img');
                img.src = `/static/imgs_sections/${percentage}/${name}`
            });
    }

    document.getElementById("filename").addEventListener("change", function() {
        const selectedFile = this.value;
        fetchImageSz(selectedFile, image_percentage);
    });

    document.addEventListener("DOMContentLoaded", function () {
        const select = document.getElementById("filename");
        const selectedFile = select.value;
        fetchImageSz(selectedFile, image_percentage);
    });

    const image = document.getElementById('main-img');
    const selectionBox = document.getElementById('selectionBox');
    let startX, startY;
    
    function moveInfoBox(x, y, w, h){
        // selectionInfo.style.display = "block";
        const rect = image.getBoundingClientRect();
        const margin = 10;
        const infoBoxWidth = selectionInfo.offsetWidth;
        let infoLeft = x + w + margin + rect.left;
        if (infoLeft + infoBoxWidth > window.innerWidth) {
            infoLeft = x + rect.left - infoBoxWidth - margin;
        }
        selectionInfo.style.left = `${infoLeft}px`;
        selectionInfo.style.top = `${y + rect.top}px`;
        // selectionInfo.style.left = (x + w + 10 + rect.left) + 'px';
        // selectionInfo.style.top = (y + rect.top) + 'px';
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

            const filename = document.getElementById("filename").value;
            ruler = metadata[filename].metrics.ruler;

            const ratio = 4 * ruler.mm / ruler.px;
            const px2mm = x => Math.round(x * ratio * 100)/100;
            const px2mm_sq = x => Math.round(x * ratio * ratio * 100)/100;

            selectionInfo.innerHTML = `${px2mm(Math.round(w * scaleX))}mm x ${px2mm(Math.round(h * scaleY))}mm — ${px2mm_sq(Math.round(w * h * scaleX * scaleY))} mm²`;
            selectionInfo.innerHTML += `<br/>${(Math.round(w * scaleX))}px x ${(Math.round(h * scaleY))}px — ${area_px} px²`;
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
                }
                if (task.state == "Requested") {
                    document.getElementById("everything").classList.add("hidden");
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
        ruler = metadata[filename].metrics.ruler;
        const ratio = 4 * ruler.mm / ruler.px;
        const mm2 = (slider.value * ratio * ratio).toFixed(4);
        sliderValueDisplay.textContent = slider.value + "px² = " + mm2 + "mm²";
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
