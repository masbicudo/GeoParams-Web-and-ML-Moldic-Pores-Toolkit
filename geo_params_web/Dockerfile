FROM python:3.12-slim

ENV PDM_CHECK_UPDATE=false \
    PDM_USE_VENV=true \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libgl1 \
        libglib2.0-0 \
        supervisor \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "pdm==2.26.1"

WORKDIR /workspace/geo_params_web

COPY geo_params_web/pyproject.toml geo_params_web/pdm.lock ./
RUN pdm sync --prod --no-editable

COPY geo_params_web/ ./
COPY user_params_porosity/measure_porosity_from_params.py /workspace/user_params_porosity/measure_porosity_from_params.py
COPY geo_params_web/supervisord.conf /etc/supervisor/conf.d/geo-params-web.conf
COPY geo_params_web/streamlit.toml /root/.streamlit/config.toml

EXPOSE 5000 8501

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]
