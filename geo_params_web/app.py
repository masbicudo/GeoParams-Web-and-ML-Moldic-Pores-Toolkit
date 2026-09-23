from contextlib import contextmanager
from flask import Flask, make_response, redirect, render_template, request, send_file, send_from_directory, url_for, flash
from flask import jsonify
from typing import Literal, Sequence, Union, Tuple, Any
import numpy as np
import cv2
import datetime as dt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import os
from libs import images
from libs.web_helpers import ensure_session, expect_session, get_session, may_have_session, save_session, load_session
from libs.web_helpers import setup_progress, progress_step, check_timeout
import threading
import json
import localizable_resources as lr
from libs.upload_datasets import (
    DatasetError,
    create_dataset,
    delete_dataset,
    find_image,
    list_datasets,
    load_dataset,
    preview_path,
    update_image_metadata,
)
from libs.porosity_tool import (
    PorosityToolError,
    load_result,
    parameter_sets,
    result_image_path,
)
from libs.porosity_jobs import get_porosity_job, submit_porosity_job

from dotenv import load_dotenv
load_dotenv(".env", override=False)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY')
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "1024")) * 1024 * 1024
# app.config['TEMPLATES_AUTO_RELOAD'] = True

if not app.secret_key:
    raise Exception(f"You must set a secret key")

IMAGE_SCALE = 12.5
PARAMETERS_GRID = 32

global_lock = threading.Lock()

class ProcessToken:
    def __init__(self, state: str = "Idle"):
        self.state = state

log_indent_level = 0
def log(session_id: str, message: str):
    session = get_session(session_id)
    counter = session["counter"] if session else "no-session"
    if session is not None:
        os.makedirs(f"log/{session_id}/{counter}", exist_ok=True)
        with open(f"log/{session_id}/{counter}/log.txt", "a") as f:
            indent = "    " * log_indent_level
            f.write(indent + f"[{dt.datetime.now()}] {message}\n")

@contextmanager
def log_indent():
    global log_indent_level
    log_indent_level += 1  # increase indentation
    try:
        yield
    finally:
        log_indent_level -= 1  # restore indentation

def args_dict_to_str(args: Union[dict, Sequence, None]) -> str:
    if args is None:
        return ""
    if isinstance(args, dict):
        items = [f"{k}={repr(v)}" for k, v in args.items()]
        return ", ".join(items)
    elif isinstance(args, (list, tuple)):
        items = [repr(v) for v in args]
        return ", ".join(items)
    else:
        return str(args)

@contextmanager
def log_function(session_id: str, function_name: str, args: Union[dict, Sequence, None] = None):
    args_str = args_dict_to_str(args)
    log(session_id, f"Enter: {function_name}({args_str})")
    global log_indent_level
    log_indent_level += 1
    try:
        yield
    finally:
        log_indent_level -= 1
        log(session_id, f"Exit: {function_name}({args_str})")

def set_task_error(task, error_message: str):
    task.update({
            "arguments": None,
            "cache": None,
            "alive_tag": None,
            "state": "Error",
            "timeout": None,
            "progress": None,  # available when task state is "Requested"
            "result": None, # available when task state is "Done"
            "error": error_message, # available when task state is "Error"
        })

def set_task_done(task, result: Any):
    task.update({
            "arguments": None,
            "cache": None,
            "alive_tag": None,
            "state": "Done",
            "timeout": None,
            "progress": None,
            "result": result,
            "error": None,
        })

def get_task_data_for_serialization(task: dict) -> dict:
    new_obj = {}
    new_obj.update(task)
    del new_obj["alive_tag"]  # Do not serialize the alive_tag object
    del new_obj["cache"]  # Do not serialize the cache
    return new_obj

def task_executor(session_id: str, task_name: str,
                  count_timeouts: int = 0,
                  arguments: Sequence | None = None,
                  keep_cache=False):
    
    with log_function(session_id, "task_executor", {"session_id": session_id, "task_name": task_name,
                                                    "count_timeouts": count_timeouts,
                                                    "arguments": arguments,
                                                    "keep_cache": keep_cache}):
        session = get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        with global_lock:
            session.setdefault("tasks", {})

            new_task_specs = {
                "arguments": arguments,
                "cache": {},
                "alive_tag": None,
                "state": "Requested",
                "timeout": None,
                "progress": None, # available when task state is "Requested"
                "result": None, # available when task state is "Done"
                "error": None, # available when task state is "Error"
            }
            
            task_exists = task_name in session["tasks"] and session["tasks"][task_name] is not None
            if task_exists:
                # If there is a task with the given name, and there are new arguments,
                # cancel previous process and reset the task with new arguments
                task = session["tasks"][task_name]
                if arguments is not None:
                    if task["alive_tag"] is not None:
                        task["alive_tag"].state = "Canceled"
                    if keep_cache:
                        new_task_specs["cache"] = task["cache"]
                    task.update(new_task_specs)
            else:
                if arguments is None:
                    raise ValueError(f"Task {task_name} requires arguments")
                task = new_task_specs
                session["tasks"][task_name] = task

            log(session_id, f"Task state: ({args_dict_to_str(task)})")
            
            # If the task is not running, start it (if it is not done already)
            not_alive = task["alive_tag"] is None
            if (not_alive):
                log(session_id, f"Flags: (not_alive={not_alive})")
                thread_args = (session_id, count_timeouts)
                args_str = args_dict_to_str(thread_args)
                log(session_id, f"Starting task thread {task_name} with {args_str}")
                function_ref = globals()[task_name]
                def task_thread_function():
                    function_ref(session_id, count_timeouts)
                thread = threading.Thread(target=task_thread_function)
                thread.start()

initial_image_setup_lock = threading.Lock()
def initial_image_setup(session_id, count_timeouts: int = 0):
#   with initial_image_setup_lock:
    if count_timeouts > 3:
        raise ValueError("Too many timeouts, aborting initial image setup.")
    
    session = get_session(session_id)

    if session is None:
        raise ValueError(f"Session {session_id} not found")

    with global_lock:
        task = session["tasks"].get("initial_image_setup")
        if task is None:
            raise ValueError(f"Task 'initial_image_setup' not found in session {session_id}")
        
        if task["state"] != "Requested":
            log(session_id, "Task 'initial_image_setup' is finished already, skipping.")
            return
        
        thresh = task["arguments"][0]
        cache = task["cache"]
        
        if "options" not in session:
            session["options"] = {}
        
        session["options"]["initial_image_setup.min_pore_size"] = thresh

        task["alive_tag"] = process_token = ProcessToken()
        process_token.state = "Running"


    # Save the base image to output folder
    path = f"static/output/{session_id}/{session['counter']}/cropped.jpg"
    base_image = cv2.imread(path)

    os.makedirs(f"static/output/{session_id}/{session['counter']}", exist_ok=True)

    timeout = 10 if os.getenv('FLASK_ENV') != "development" else 60*60*24
    set_total_steps = lambda steps: setup_progress(
        session_id, "initial_image_setup", steps,
        timeout=timeout)
    def do_step():
        if process_token.state == "Timeout":
            task_executor(session_id, "initial_image_setup",
                            count_timeouts + 1, [thresh],
                            keep_cache=True)
            return False
        if process_token.state != "Running": return False
        progress_step(session_id, "initial_image_setup")
        return True
    map_img, map_images = images.get_images(base_image,
                                            step=256//PARAMETERS_GRID,
                                            thresh=thresh,
                                            set_total_steps=set_total_steps,
                                            do_step=do_step)

    # check if canceled
    if (map_img is None
        or map_images is None
        or process_token.state != "Running"):
        return

    cv2.imwrite(
        f"static/output/{session_id}/{session['counter']}/main_image.png",
        map_img
        )
    
    # Compute dimensions
    tile_h, tile_w = map_images[0][0].shape[:2]
    rows = len(map_images)
    cols = len(map_images[0])

    # Create a large black image
    stitched_img = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

    for y, row in enumerate(map_images):
        for x, tile in enumerate(row):
            y_start = y * tile_h
            x_start = x * tile_w
            if process_token.state != "Running":
                return
            stitched_img[y_start:y_start + tile_h, x_start:x_start + tile_w] = tile

    # Save the stitched image
    cv2.imwrite(f"static/output/{session_id}/{session['counter']}/stitched_tiles.png", stitched_img)
    
    session_thresh = session["options"]["initial_image_setup.tile_shape"] = [tile_h, tile_w]

    process_token.state = "Done"
    set_task_done(task, {"tile_shape": [tile_h, tile_w]})

@app.context_processor
def inject_endpoint():
    from flask import request
    return dict(
            current_endpoint=request.endpoint,
            lr=lr,
        )

@app.route('/static/cached/<path:filename>')
def bootstrap_static(filename):
    path = os.path.join(app.root_path, 'static')
    response = make_response(send_from_directory(path, filename))
    # Cache for 30 days
    response.headers['Cache-Control'] = 'public, max-age=2592000'
    return response

@app.route('/')
@may_have_session
def index(session_id):
    session = get_session(session_id)
    
    user_name = None
    if session is not None:
        renew = bool(request.args.get('renew', None))
        if renew:
            session["counter"] += 1
            return redirect(url_for("index", session_id=session_id, **(request.view_args or {})))
    
    user_name = lr.str.anonymous_user("participant")
    
    return render_template('actions.html',
                           has_session=session is not None,
                           session_id=session_id,
                           session=session,
                           user_name=user_name,
                           )


@app.route('/about-data')
def about_data():
    return render_template('about_data.html')


@app.route('/porosity', methods=['GET', 'POST'])
def porosity_calculator():
    session_id = request.args.get('session_id')
    if request.method == 'GET':
        return render_template(
            'porosity.html',
            session_id=session_id,
            parameter_sets=parameter_sets(),
            result=None,
        )

    try:
        job = submit_porosity_job(
            request.form.get('dataset_id', ''),
            request.files.get('image'),
            bootstrap=request.form.get('bootstrap') == 'yes',
        )
    except (PorosityToolError, DatasetError, OSError, ValueError) as exc:
        flash(str(exc), 'danger')
        return redirect(url_for('porosity_calculator', session_id=session_id))

    return redirect(url_for('porosity_job', job_id=job['id'], session_id=session_id))


@app.route('/porosity/jobs/<job_id>')
def porosity_job(job_id):
    session_id = request.args.get('session_id')
    try:
        job = get_porosity_job(job_id)
    except PorosityToolError as exc:
        flash(str(exc), 'danger')
        return redirect(url_for('porosity_calculator', session_id=session_id))
    return render_template(
        'porosity_progress.html',
        session_id=session_id,
        job=job,
    )


@app.route('/porosity/jobs/<job_id>/status')
def porosity_job_status(job_id):
    try:
        job = get_porosity_job(job_id)
    except PorosityToolError as exc:
        return jsonify({'status': 'error', 'error': str(exc)}), 404
    return jsonify(job)


@app.route('/porosity/results/<run_id>')
def porosity_result(run_id):
    session_id = request.args.get('session_id')
    try:
        result = load_result(run_id)
    except PorosityToolError as exc:
        flash(str(exc), 'danger')
        return redirect(url_for('porosity_calculator', session_id=session_id))
    return render_template(
        'porosity.html',
        session_id=session_id,
        parameter_sets=parameter_sets(),
        result=result,
    )


@app.route('/porosity/results/<run_id>/images/<kind>')
def porosity_result_image(run_id, kind):
    try:
        return send_file(result_image_path(run_id, kind))
    except PorosityToolError:
        return "Analysis image not found", 404


@app.route('/user_id', methods=['GET'])
@ensure_session
def user_id(session_id):
    session = get_session(session_id)
    if session is None:
        raise ValueError(f"Session {session_id} not found")
    
    user = session.setdefault("options", {}).setdefault("user", {})
    # NOTE:
    # Personal identifiers are not collected or stored in the public release.
    user["email"] = None
    user["name"] = None
    
    return redirect(url_for("index", session_id=session_id))

@app.route('/userinfo', methods=['GET', 'POST'])
@ensure_session
def userinfo(session_id):
    session = get_session(session_id)
    if session is None:
        raise ValueError(f"Session {session_id} not found")
    if request.method == 'GET':
        user = session.setdefault("options", {}).setdefault("user", {})
        experience = user.setdefault("experience", 0)
        return render_template('userinfo.html',
                               session_id=session_id,
                               user=user)
    if request.method == 'POST':
        user = session["options"]["user"]
        experience = user["experience"] = int(request.form.get('experience', 0))
        if experience == 0:
            flash(lr.str.fill_all_fields, 'error')
            return redirect(url_for("userinfo",
                                    session_id=session_id))
        return redirect(url_for("image_select",
                                session_id=session_id))

    # This should never be reached, but ensures all paths return
    return "Method not allowed", 405

@app.route('/params_select', methods=['GET', 'POST'])
@ensure_session
def params_select(session_id):
    with log_function(session_id, "params_select", {}):
        session = get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        session_min_pore_size = session.setdefault("options", {}).setdefault("initial_image_setup.min_pore_size", 480)
        tile_shape = session.setdefault("options", {}).get("initial_image_setup.tile_shape", [])
        points = session.setdefault("options", {}).setdefault("params_select.clicked_points", [])
        priority = session.setdefault("options", {}).setdefault("params_select.priority", "none")
        log(session_id, f"request.method = {request.method}")
        if request.method == 'GET':
            return render_template('params_select.html', session_id=session_id,
                                min_pore_size=session_min_pore_size,
                                clicked_points=points,
                                tile_shape=tile_shape,
                                filename=session.get("options", {})["image_select.filename"],
                                mm_per_pixel=session.get("options", {}).get("image_select.mm_per_pixel"),
                                counter=session["counter"],
                                priority=priority,
                                lr=lr,
                            )
            
        elif request.method == 'POST':
            data = request.get_json()
            if "end_reason" in data:
                end_reason = data["end_reason"]
                log(session_id, f"end_reason = {end_reason}")
                try:
                    with global_lock:
                        if end_reason == "cancel":
                            state = "Cancel"
                            log(session_id, f"steate = {state}")
                            session["options"]["params_select.state"] = state
                            return jsonify({"status": "canceled"}), 200
                        elif end_reason == "done":
                            errors = []
                            
                            log(session_id, f"1")
                            # priority field
                            priority = data.get("priority", "none")
                            if priority in ["connectivity", "size", "shape"]:
                                session["options"]["params_select.priority"] = priority
                            else:
                                errors.append(lr.str.priority_field_is_missing)
                                
                            log(session_id, f"2")
                            # if any errors, return them
                            if len(errors) > 0:
                                state = "Error"
                                log(session_id, f"steate = {state}")
                                session["options"]["params_select.state"] = state
                                flash("\n".join(errors), 'danger')
                                return jsonify({"status": "error"}), 200
                            
                            log(session_id, f"3")
                            # if everything is ok, return success
                            flash(lr.str.data_saved, 'success')
                            log(session_id, f"4")
                            state = "Done"
                            log(session_id, f"steate = {state}")
                            session["options"]["params_select.state"] = state
                            return jsonify({"status": "done"}), 200
                        else:
                            return jsonify({"error": "Unknown end reason"}), 400
                finally:
                    # Save the session options after processing the end reason
                    data = session["options"]
                    with open(f"static/output/{session_id}/{session['counter']}/options.json", "w") as f:
                        json.dump(data, f, indent=2)
                    state = session["options"]["params_select.state"]
                    with open(f"static/output/{session_id}/{session['counter']}/params_select.state={state}", "w") as f:
                        pass
            
            min_pore_size = int(data.get('min_pore_size', 480))
            
            task_executor(session_id, "initial_image_setup",
                        0, [min_pore_size])

            print("Received min_pore_size:", min_pore_size)
            return jsonify({'min_pore_size': min_pore_size})

        # This should never be reached, but ensures all paths return
        return "Method not allowed", 405

# NOTE:
# This endpoint previously collected free-text user feedback during
# supervised study sessions. It is disabled in the public release to
# avoid collecting potentially identifiable information.
@app.route('/end_review', methods=['GET'])
@ensure_session
def end_review(session_id):
    session = get_session(session_id)
    if session is None:
        raise ValueError(f"Session {session_id} not found")

    return redirect(url_for("index", session_id=session_id, renew=True))

def getImageFiles():
    IMAGE_FILES_DIR = "static/imgs_sections"
    files = [f for f in os.listdir(IMAGE_FILES_DIR)
                if os.path.isfile(os.path.join(IMAGE_FILES_DIR, f))
                and (f.lower().endswith('.jpg') or f.lower().endswith('.jpeg'))]
    return files


def builtin_image_choices() -> list[dict]:
    files = getImageFiles()
    with open("static/imgs_sections/metadata.json", encoding="utf-8") as fp:
        metadata = json.load(fp)
    choices = []
    for filename in files:
        ruler = metadata.get(filename, {}).get("metrics", {}).get("ruler")
        mm_per_pixel = None
        if ruler and ruler.get("px") and ruler.get("mm"):
            # Preserve the conversion used by the published application for
            # its 12.5% processing images.
            mm_per_pixel = 4.0 * float(ruler["mm"]) / float(ruler["px"])
        choices.append(
            {
                "id": filename,
                "label": filename,
                "preview_url": url_for(
                    "static", filename=f"imgs_sections/{IMAGE_SCALE}/{filename}"
                ),
                "mm_per_pixel": mm_per_pixel,
            }
        )
    return choices


def custom_image_choices(dataset: dict, session_id: str) -> list[dict]:
    choices = []
    for image in dataset["images"]:
        calibration = image.get("calibration")
        mm_per_pixel = None
        if calibration:
            original_um_per_pixel = calibration.get("micrometers_per_pixel")
            preview_scale = image.get("preview_scale")
            if original_um_per_pixel and preview_scale:
                mm_per_pixel = float(original_um_per_pixel) / float(preview_scale) / 1000.0
        choices.append(
            {
                "id": image["id"],
                "label": image.get("display_name") or image["original_filename"],
                "preview_url": url_for(
                    "custom_image_preview",
                    dataset_id=dataset["id"],
                    image_id=image["id"],
                    session_id=session_id,
                ),
                "mm_per_pixel": mm_per_pixel,
            }
        )
    return choices


@app.route('/datasets', methods=['GET', 'POST'])
@ensure_session
def datasets(session_id):
    if request.method == 'POST':
        try:
            dataset = create_dataset(
                request.form.get("name", ""),
                request.files.getlist("images"),
            )
        except DatasetError as exc:
            flash(str(exc), 'danger')
            return redirect(url_for("datasets", session_id=session_id))
        flash(f"Dataset '{dataset['name']}' created.", 'success')
        return redirect(
            url_for("dataset_detail", dataset_id=dataset["id"], session_id=session_id)
        )
    return render_template(
        'datasets.html',
        session_id=session_id,
        datasets=list_datasets(),
        max_upload_mb=app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
    )


@app.route('/datasets/<dataset_id>')
@ensure_session
def dataset_detail(dataset_id, session_id):
    try:
        dataset = load_dataset(dataset_id)
    except DatasetError as exc:
        flash(str(exc), 'danger')
        return redirect(url_for("datasets", session_id=session_id))
    return render_template(
        'dataset_detail.html', dataset=dataset, session_id=session_id
    )


@app.route('/datasets/<dataset_id>/images/<image_id>/metadata', methods=['POST'])
@ensure_session
def dataset_image_metadata(dataset_id, image_id, session_id):
    try:
        pixel_distance = request.form.get("pixel_distance_preview", "").strip()
        physical_distance = request.form.get("physical_distance", "").strip()
        unit = request.form.get("unit", "").strip() or None
        update_image_metadata(
            dataset_id,
            image_id,
            request.form.get("display_name", ""),
            float(pixel_distance) if pixel_distance else None,
            float(physical_distance) if physical_distance else None,
            unit,
        )
        flash("Image metadata saved.", 'success')
    except (DatasetError, ValueError) as exc:
        flash(str(exc), 'danger')
    return redirect(
        url_for("dataset_detail", dataset_id=dataset_id, session_id=session_id)
    )


@app.route('/datasets/<dataset_id>/delete', methods=['POST'])
@ensure_session
def dataset_delete(dataset_id, session_id):
    try:
        dataset = load_dataset(dataset_id)
        confirmation = request.form.get("confirmation", "")
        if confirmation != dataset["name"]:
            raise DatasetError("Type the dataset name exactly to confirm deletion.")
        delete_dataset(dataset_id)
        session = get_session(session_id)
        if session and session.get("options", {}).get("image_select.dataset_id") == dataset_id:
            session["options"]["image_select.dataset_id"] = "builtin"
        flash(f"Dataset '{dataset['name']}' deleted.", 'success')
    except DatasetError as exc:
        flash(str(exc), 'danger')
    return redirect(url_for("datasets", session_id=session_id))


@app.route('/datasets/<dataset_id>/previews/<image_id>')
@ensure_session
def custom_image_preview(dataset_id, image_id, session_id):
    try:
        dataset = load_dataset(dataset_id)
        image = find_image(dataset, image_id)
        return send_file(preview_path(dataset_id, image), mimetype="image/jpeg")
    except DatasetError as exc:
        return str(exc), 404

@app.route('/image_select', methods=['GET', 'POST'])
@ensure_session
def image_select(session_id):
    if request.method == 'GET':
        session = get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        options = session.setdefault("options", {})
        requested_dataset_id = request.args.get("dataset_id")
        current_dataset_id = requested_dataset_id or options.get(
            "image_select.dataset_id", "builtin"
        )
        dataset = None
        try:
            if current_dataset_id == "builtin":
                images_available = builtin_image_choices()
                dataset_name = "Publication thin sections"
            else:
                dataset = load_dataset(current_dataset_id)
                images_available = custom_image_choices(dataset, session_id)
                dataset_name = dataset["name"]
        except DatasetError as exc:
            flash(str(exc), 'danger')
            current_dataset_id = "builtin"
            images_available = builtin_image_choices()
            dataset_name = "Publication thin sections"
        options["image_select.dataset_id"] = current_dataset_id

        if not images_available:
            return (
                "No processed images were found in the selected dataset.",
                400,
            )

        image_id = options.get("image_select.image_id", images_available[0]["id"])
        if image_id not in {item["id"] for item in images_available}:
            image_id = images_available[0]["id"]
        return render_template('image_select.html', session_id=session_id,
                               images_available=images_available,
                               current_image_id=image_id,
                               current_dataset_id=current_dataset_id,
                               dataset_name=dataset_name,
                               datasets=list_datasets(),
                               area_x="0",
                               area_y="0",
                               area_w="0",
                               area_h="0",
                               )
    elif request.method == 'POST':
        session = get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        
        # deleting associated options
        session.get("options", {}).pop("initial_image_setup.min_pore_size", None)
        session.get("options", {}).pop("initial_image_setup.tile_shape", None)
        session.get("options", {}).pop("params_select.clicked_points", None)

        dataset_id = request.form.get("dataset_id", "builtin")
        image_id = request.form["image_id"]
        x = int(request.form["x"])
        y = int(request.form["y"])
        w = int(request.form["w"])
        h = int(request.form["h"])

        if dataset_id == "builtin":
            choices = builtin_image_choices()
            selected = next((item for item in choices if item["id"] == image_id), None)
            if selected is None:
                return "Image not found", 404
            input_path = f"static/imgs_sections/{IMAGE_SCALE}/{image_id}"
        else:
            try:
                dataset = load_dataset(dataset_id)
                image_record = find_image(dataset, image_id)
                selected = next(
                    item for item in custom_image_choices(dataset, session_id) if item["id"] == image_id
                )
                input_path = str(preview_path(dataset_id, image_record))
            except (DatasetError, StopIteration):
                return "Image not found", 404

        options = session.setdefault("options", {})
        options["image_select.dataset_id"] = dataset_id
        options["image_select.image_id"] = image_id
        options["image_select.filename"] = selected["label"]
        options["image_select.mm_per_pixel"] = selected["mm_per_pixel"]

        if h == 0 or w == 0:
            flash("Select a region of interest before continuing.", "danger")
            return redirect(
                url_for("image_select", session_id=session_id, dataset_id=dataset_id)
            )

        output_path = f"static/output/{session_id}/{session['counter']}/cropped.jpg"

        # Load and crop
        image = cv2.imread(input_path)
        if image is None:
            return "Image not found", 404

        cropped = image[y:y+h, x:x+w]

        # Save cropped image
        os.makedirs(f"static/output/{session_id}/{session['counter']}", exist_ok=True)
        cv2.imwrite(output_path, cropped)

        task_executor(session_id, "initial_image_setup",
                      0, [480])
        
        return redirect(url_for("params_select", session_id=session_id))

    # This should never be reached, but ensures all paths return
    return "Method not allowed", 405

@app.route('/make_thin_section')
def make_thin_section():
    name = request.args.get('name')
    percentage = request.args.get('percentage', 0)
    if os.path.isfile(f"static/imgs_sections/{percentage}/{name}"):
        return jsonify({'error': 'Already exists'})
    os.makedirs(f"static/imgs_sections/{percentage}", exist_ok=True)
    images.resize_image_file(
            f"static/imgs_sections/{name}",
            f"static/imgs_sections/{percentage}/{name}",
            float(percentage),
        )
    return jsonify({'value': 'Ok'})

@app.route('/hover')
@expect_session
def hover(session_id):
    session = get_session(session_id)
    if session is None:
        raise ValueError(f"Session {session_id} not found")
    x = int(request.args.get('x', 0))
    y = int(request.args.get('y', 0))

    filename = f"static/output/{session_id}/{session['counter']}/x={x},y={y}.png"

    with open(filename, 'rb') as f:
        buf = io.BytesIO(f.read())
        buf.seek(0)
        return send_file(buf, mimetype='image/png')

@app.route('/get_task_info/<name>')
@expect_session
def get_task_info(session_id, name):

    with log_function(session_id, "get_task_info", {"name": name}):

        # If there is a task with the given name, ensure it is running
        task_executor(session_id, name, 0)
        
        # We check the timeout of a task whenever its info is requested
        check_timeout(session_id, name)
        
        session = get_session(session_id)
        if session is None:
            return jsonify({'error': 'Session ID not found'}), 400
        with global_lock:
            if name in session["tasks"]:
                return jsonify({"task": get_task_data_for_serialization(session["tasks"][name])})
            return jsonify({'error': 'Task not found'}), 400


@app.route('/add_point', methods=['POST'])
@expect_session
def add_point(session_id):
    data = request.get_json()
    x = int(data.get("x"))
    y = int(data.get("y"))

    session = get_session(session_id)
    if session is None:
        raise ValueError(f"Session {session_id} not found")
    clicked_points = session.setdefault("options", {}).setdefault("params_select.clicked_points", [])

    # Prevent duplicates (optional)
    if any(pt["x"] == x and pt["y"] == y for pt in clicked_points):
        return jsonify({"status": "duplicate", "point": {"x": x, "y": y}}), 200

    # Add point
    clicked_points.append({"x": x, "y": y})

    # Save to file
    min_pore_size = session.get("options", {}).get("initial_image_setup.min_pore_size", 480)
    output_file = f"static/output/{session_id}/{session['counter']}/clicked_points.json"
    with open(output_file, "w") as f:
        json.dump({
            "min_pore_size": min_pore_size,
            "points": clicked_points
        }, f, indent=2)

    return jsonify({"status": "ok", "added": {"x": x, "y": y}})


@app.route('/delete_point', methods=['POST'])
@expect_session
def delete_point(session_id):
    data = request.get_json()
    x = int(data.get("x"))
    y = int(data.get("y"))

    session = get_session(session_id)
    if session is None:
        raise ValueError(f"Session {session_id} not found")
    clicked_points = session.setdefault("options", {}).setdefault("params_select.clicked_points", [])

    # Check if point exists
    original_len = len(clicked_points)
    clicked_points = [pt for pt in clicked_points if not (pt["x"] == x and pt["y"] == y)]
    deleted = (len(clicked_points) < original_len)
    session["clicked_points"] = clicked_points

    # Save updated list
    min_pore_size = session.get("options", {}).get("initial_image_setup.min_pore_size", 480)
    output_file = f"static/output/{session_id}/{session['counter']}/clicked_points.json"
    with open(output_file, "w") as f:
        json.dump({
            "min_pore_size": min_pore_size,
            "points": clicked_points
        }, f, indent=2)

    if deleted:
        return jsonify({
            "status": "ok",
            "deleted": {"x": x, "y": y}
        })
    else:
        return jsonify({
            "status": "not_found",
            "message": "Point not in list",
            "requested": {"x": x, "y": y}
        }), 404


if __name__ == "__main__":
    load_session()
    try:
        if os.getenv("FLASK_ENV") == "development":
            app.run(debug=True, use_reloader=False)
        else:
            app.run(host="0.0.0.0", port=5000)
    finally:
        # Only save session in the actual serving process,
        # not in the Flask reloader bootstrap.
        if os.getenv("WERKZEUG_RUN_MAIN") == "true":
            save_session()
