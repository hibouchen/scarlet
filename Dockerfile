FROM python:3.12.3-slim-bookworm

ENV MPLBACKEND=Agg \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        gosu \
        libegl1 \
        libgl1 \
        libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --shell /bin/bash jovyan

WORKDIR /opt/scarlet

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install --no-cache-dir . "jupyterhub==6.0.0"

COPY notebooks/tutorial.ipynb /opt/scarlet/tutorial.ipynb
COPY deployment/jupyterhub/start-scarlet-singleuser.sh /usr/local/bin/start-scarlet-singleuser

RUN chmod 755 /usr/local/bin/start-scarlet-singleuser \
    && mkdir -p /home/jovyan/work \
    && chown -R jovyan:jovyan /home/jovyan

USER jovyan
WORKDIR /home/jovyan/work

CMD ["jupyterhub-singleuser"]
