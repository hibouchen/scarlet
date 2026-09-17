from __future__ import annotations

import os


def _csv_env(name: str) -> set[str]:
    return {value.strip() for value in os.environ.get(name, "").split(",") if value.strip()}


auth_mode = os.environ.get("JUPYTERHUB_AUTH_MODE", "github").strip().lower()
allowed_users = _csv_env("JUPYTERHUB_ALLOWED_USERS")
admin_users = _csv_env("JUPYTERHUB_ADMIN_USERS")

if not allowed_users:
    raise RuntimeError("JUPYTERHUB_ALLOWED_USERS must contain at least one GitHub login")
if not admin_users:
    raise RuntimeError("JUPYTERHUB_ADMIN_USERS must contain at least one GitHub login")

c.JupyterHub.bind_url = "http://:8000"
c.JupyterHub.hub_bind_url = "http://:8081"
c.JupyterHub.cookie_secret = bytes.fromhex(os.environ["JUPYTERHUB_COOKIE_SECRET"])
c.JupyterHub.db_url = "sqlite:////srv/jupyterhub/jupyterhub.sqlite"
c.JupyterHub.admin_users = admin_users
c.Authenticator.allowed_users = allowed_users
c.Authenticator.allow_all = False

if auth_mode == "github":
    client_id = os.environ.get("JUPYTERHUB_GITHUB_CLIENT_ID", "")
    client_secret = os.environ.get("JUPYTERHUB_GITHUB_CLIENT_SECRET", "")
    callback_url = os.environ.get("JUPYTERHUB_OAUTH_CALLBACK_URL", "")
    if not client_id or not client_secret or not callback_url:
        raise RuntimeError(
            "GitHub OAuth requires JUPYTERHUB_GITHUB_CLIENT_ID, "
            "JUPYTERHUB_GITHUB_CLIENT_SECRET, and JUPYTERHUB_OAUTH_CALLBACK_URL"
        )
    from oauthenticator.github import GitHubOAuthenticator

    c.JupyterHub.authenticator_class = GitHubOAuthenticator
    c.GitHubOAuthenticator.client_id = client_id
    c.GitHubOAuthenticator.client_secret = client_secret
    c.GitHubOAuthenticator.oauth_callback_url = callback_url
elif auth_mode == "dummy":
    password = os.environ.get("JUPYTERHUB_DUMMY_PASSWORD", "")
    if not password:
        raise RuntimeError("Dummy authentication requires JUPYTERHUB_DUMMY_PASSWORD")
    from jupyterhub.auth import DummyAuthenticator

    c.JupyterHub.authenticator_class = DummyAuthenticator
    c.DummyAuthenticator.password = password
else:
    raise RuntimeError("JUPYTERHUB_AUTH_MODE must be either 'github' or 'dummy'")

c.JupyterHub.spawner_class = "dockerspawner.DockerSpawner"
c.DockerSpawner.image = os.environ.get("SCARLET_NOTEBOOK_IMAGE", "scarlet-notebook:0.2.4")
c.DockerSpawner.network_name = os.environ.get("DOCKER_NETWORK_NAME", "scarlet-jupyterhub")
c.DockerSpawner.use_internal_ip = True
c.DockerSpawner.remove = True
c.DockerSpawner.extra_create_kwargs = {"user": "root"}
c.DockerSpawner.volumes = {
    "scarlet-user-{username}": {"bind": "/home/jovyan/work", "mode": "rw"},
}

shared_data_dir = os.environ.get("SCARLET_SHARED_DATA_DIR", "").strip()
if shared_data_dir:
    c.DockerSpawner.volumes[shared_data_dir] = {
        "bind": "/home/jovyan/shared-data",
        "mode": "ro",
    }

c.Spawner.cmd = ["/usr/local/bin/start-scarlet-singleuser"]
c.Spawner.default_url = "/lab/tree/tutorial.ipynb"
c.Spawner.environment = {"SCARLET_WORK_DIR": "/home/jovyan/work"}
c.Spawner.hub_connect_url = "http://hub:8081"
c.Spawner.http_timeout = 120
c.Spawner.ip = "0.0.0.0"
c.Spawner.start_timeout = 120
