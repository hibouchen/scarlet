FROM jupyterhub/jupyterhub:6.0.0

RUN python -m pip install --no-cache-dir \
    "dockerspawner==14.0.0" \
    "oauthenticator==17.4.0"

COPY jupyterhub_config.py /srv/jupyterhub/jupyterhub_config.py
