# SCARLET JupyterHub

This deployment gives each allowed user a browser-based JupyterLab server using the same SCARLET container image. It is intended for a Linux host with Docker Engine and Docker Compose.

The current Qt viewer is not exposed by this deployment. It requires a dedicated browser UI before it can replace the desktop viewer.

## First deployment

1. Copy `.env.example` to `.env` and replace every placeholder.
2. Create a GitHub OAuth application. Its callback URL must exactly match `JUPYTERHUB_OAUTH_CALLBACK_URL`, for example `https://hub.example.org/hub/oauth_callback`.
3. Generate the cookie secret with `openssl rand -hex 32` and set it in `.env`.
4. Build the Hub and immutable SCARLET notebook image:

   ```bash
   docker compose -f deployment/jupyterhub/compose.yml build
   ```

5. Start the Hub:

   ```bash
   docker compose -f deployment/jupyterhub/compose.yml up -d hub
   ```

6. Put a TLS reverse proxy in front of `127.0.0.1:8000` and route a domain such as `hub.example.org` to it.

The Hub is deliberately bound to the loopback interface. Do not expose it directly to the Internet without HTTPS.

The Hub needs access to the Docker socket to create user servers. Treat Hub administrators as trusted administrators of the Docker host.

## User storage and shared data

Every user receives a persistent Docker volume mounted at `/home/jovyan/work`. The container initializes its ownership on first start, then launches JupyterLab as the unprivileged `jovyan` user. The tutorial is copied there on the first start, so users can edit it without changing the source image.

Set `SCARLET_SHARED_DATA_DIR` to a host directory to mount it read-only at `/home/jovyan/shared-data` for all users. Write reduction products in each user's workspace or use a separate managed project storage area.

## Local demonstration only

For a local test without GitHub OAuth, set `JUPYTERHUB_AUTH_MODE=dummy`, provide `JUPYTERHUB_DUMMY_PASSWORD`, and use only the logins listed in `JUPYTERHUB_ALLOWED_USERS`. Never use this mode on a network-accessible server.

## Updating SCARLET

Build a new notebook image for each SCARLET release, for example `scarlet-notebook:0.2.5`, then set `SCARLET_NOTEBOOK_IMAGE` to that image and restart the Hub. In production, use an image digest rather than a mutable tag.
