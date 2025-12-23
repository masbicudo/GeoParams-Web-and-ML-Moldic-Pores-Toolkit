#!/bin/bash
FLASK_ENV="development" FLASK_SECRET_KEY="dev_flask_key" pdm run python app.py
