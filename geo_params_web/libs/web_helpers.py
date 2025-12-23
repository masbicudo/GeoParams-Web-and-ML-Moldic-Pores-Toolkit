import json
import os
import threading
from typing import Any
from flask import request, redirect, url_for, jsonify
import uuid
from functools import wraps
import datetime as dt

# In-memory session store (not safe for production)
session_store = {}

global_lock = threading.Lock()

def new_session_data():
    return {
        'counter': 0,
        'progress': {
                'name': "",
                'step': 0,
                'total_steps': 1,
            },
        'processes': {}
        }

def setup_progress(session_id, task_name, total_steps, timeout=60*60*24):
    with global_lock:
        session = session_store[session_id]
        task = session["tasks"][task_name]
        progress = task["progress"]
        if progress is None:
            progress = {
                "step": 0,
                "total_steps": total_steps,
                "last_update": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            task["progress"] = progress
        progress["step"] = 0
        progress["total_steps"] = total_steps
        now = dt.datetime.now()
        progress["last_update"] = now.strftime("%Y-%m-%d %H:%M:%S")
        task["timeout"] = timeout

def progress_step(session_id, task_name):
    with global_lock:
        session = session_store[session_id]
        task = session["tasks"][task_name]
        progress = task["progress"]
        progress["step"] += 1
        now = dt.datetime.now()
        progress["last_update"] = now.strftime("%Y-%m-%d %H:%M:%S")

def check_timeout(session_id, task_name):
    with global_lock:
        session = session_store[session_id]
        task = session["tasks"][task_name]
        progress = task["progress"]
        if progress is None:
            return
        last_update = progress.get("last_update", "2000-01-01 00:00:00")
        dt_last_update = dt.datetime.strptime(last_update, "%Y-%m-%d %H:%M:%S")
        now = dt.datetime.now()
        elapsed = now - dt_last_update
        alive_tag = task["alive_tag"]
        if alive_tag is not None and elapsed > dt.timedelta(seconds=task["timeout"]):
            alive_tag["state"] = "Timeout"

def save_session():
    with global_lock:
        with open("session.json", "w") as fp:
            json.dump(session_store, fp, indent=2, default=lambda o: None)

def load_session():
    global session_store
    with global_lock:
        if os.path.isfile("session.json"):
            with open("session.json", "r") as fp:
                session_store = json.load(fp)

def ensure_session(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with global_lock:
            session_id = request.args.get('session_id')
            if not session_id or session_id not in session_store:
                # Generate new session and redirect to same endpoint
                session_id = str(uuid.uuid4())
                session_store[session_id] = new_session_data()
                return redirect(url_for(request.endpoint, session_id=session_id, **request.view_args))
        # Session exists; call the wrapped function
        return func(*args, session_id=session_id, **kwargs)
    return wrapper

def expect_session(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        session_id = request.args.get('session_id')
        if not session_id or session_id not in session_store:
            return jsonify({'error': 'Invalid session ID'}), 400
        # Session exists; call the wrapped function
        return func(*args, session_id=session_id, **kwargs)
    return wrapper

def may_have_session(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        session_id = request.args.get('session_id')
        # Session exists; call the wrapped function
        return func(*args, session_id=session_id, **kwargs)
    return wrapper

def get_session(session_id) -> dict[str, Any] | None:
    if not session_id or session_id not in session_store:
        return None
    return session_store[session_id]
