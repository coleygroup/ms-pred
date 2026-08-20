# MS-Pred WebUI

Web interface for **ms-pred (ICEBERG)** spectrum retrieval and visualization.

This application is built with Flask and should be served in production using **Gunicorn behind NGINX**.

---

## Deployment Overview

Production stack:

```
Internet
   ↓
NGINX (port 80/443)
   ↓
Gunicorn (127.0.0.1:4285)
   ↓
Flask app (wsgi:app)
```

Optional read-only atlas file access:

```
SFTP client
   ↓
AsyncSSH SFTP service (port 2222)
   ↓
Atlas virtual filesystem
```

---

## Clone Repository

On the target server:

```bash
git clone git@github.com:coleygroup/ms-pred.git
cd ms-pred/webui
```

---

## Create Environment

The environment is fully defined in `environment.yml`.

```bash
mamba env create -f environment.yml
mamba activate iceberg-webui
```

To update an existing environment:

```bash
mamba env update -f environment.yml --prune
```

---

## Required Environment Variables

The application requires the following environment variables:

| Variable            | Description                           |
|--------------------|---------------------------------------|
| `FLASK_SECRET_KEY` | Secret key for session security       |
| `MSPRED_ATLAS_DIR` | Base directory of predicted MGF atlas |
| `MSPRED_JOB_DIR`   | Directory for temporary job storage   |
| `ICEBERG_USERS_FILE` | YAML file with WebUI email/password users |

The optional NIST'23 atlas is configured with:

| Variable                 | Description                                  |
|--------------------------|----------------------------------------------|
| `MSPRED_ATLAS_DIR_NIST`  | Base directory of the gated NIST'23 atlas    |
| `ICEBERG_ADMIN_EMAILS`   | Comma-separated admin safety-net email list  |

Email notifications are optional. If email is not configured, the admin
dashboard displays generated temporary passwords on-screen.

SMTP credentials must be stored only in the deployment environment or an
`EnvironmentFile` excluded from source control. The available settings are:

| Variable           | Description                                      |
|--------------------|--------------------------------------------------|
| `SMTP_HOST`        | SMTP relay hostname                              |
| `SMTP_PORT`        | SMTP relay port, defaults to `587`               |
| `SMTP_USER`        | SMTP login username                              |
| `SMTP_PASSWORD`    | SMTP login password                              |
| `SMTP_FROM`        | Sender address, defaults to `SMTP_USER`          |
| `SMTP_USE_TLS`     | Whether to use STARTTLS, defaults to `true`      |

Example:

```bash
export FLASK_SECRET_KEY="replace-with-long-random-string"
export MSPRED_ATLAS_DIR="/data/atlas"
export MSPRED_ATLAS_DIR_NIST="/data/atlas_nist"
export MSPRED_JOB_DIR="/var/lib/iceberg_jobs"
export ICEBERG_USERS_FILE="/var/lib/iceberg_users.yaml"
export ICEBERG_ADMIN_EMAILS="admin@example.com"
export SMTP_HOST="smtp.example.com"
export SMTP_PORT="587"
export SMTP_USER="iceberg@example.com"
export SMTP_PASSWORD="replace-with-an-app-password"
export SMTP_FROM="iceberg@example.com"
export SMTP_USE_TLS="true"
```

Ensure the job directory exists and has proper permissions:

```bash
sudo mkdir -p /var/lib/iceberg_jobs
sudo chown -R coley-group:coley-group /var/lib/iceberg_jobs
```

---

## Test Gunicorn Manually

From the `webui/` directory:

```bash
mamba activate iceberg-webui
gunicorn -w 2 -k gthread --threads 8 -b 0.0.0.0:4285 wsgi:app
```

Then open:

```
http://server-ip:4285
```

If the application loads correctly, proceed to the systemd setup.

---

## Install NGINX

On Ubuntu:

```bash
sudo apt update
sudo apt install nginx -y
sudo systemctl enable nginx
sudo systemctl start nginx
```

---

## Install systemd Service

Copy the service file:

```bash
sudo cp deploy/systemd-iceberg-webui.service \
  /etc/systemd/system/iceberg-webui.service
```
Please remember to replace the ``FLASK_SECRET_KEY`` and update other parameters accordingly.

Reload and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable iceberg-webui
sudo systemctl start iceberg-webui
sudo systemctl status iceberg-webui
```

---

## Install Read-Only SFTP Service

The SFTP service uses the same `ICEBERG_USERS_FILE` credentials as the WebUI.
Usernames are full email addresses. Users with role `user` see only `/public`;
roles `authorized_user` and `admin` also see `/nist23` when
`MSPRED_ATLAS_DIR_NIST` is configured.

Install or update the environment so `asyncssh` is available:

```bash
mamba env update -f environment.yml --prune
```

Generate a dedicated host key outside the repository:

```bash
ssh-keygen -t ed25519 -f /path/to/iceberg_sftp_ed25519 -N ""
```

Copy the service file:

```bash
sudo cp deploy/systemd-iceberg-sftp.service \
  /etc/systemd/system/iceberg-sftp.service
```

Edit the placeholders in `/etc/systemd/system/iceberg-sftp.service`, then reload
and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now iceberg-sftp
sudo systemctl status iceberg-sftp
```

Open TCP port `2222` in any host firewall or security group if it is blocked.

Client examples:

```bash
sftp -P 2222 -o User=user@domain.edu iceberg-ms.mit.edu
sshfs -p 2222 -o User=user@domain.edu iceberg-ms.mit.edu:/ ~/iceberg-atlas
```

The exposed virtual paths are:

| Virtual path       | Source directory                              |
|--------------------|-----------------------------------------------|
| `/public/h_plus/`  | `MSPRED_ATLAS_DIR/h_plus_out_mgf/`            |
| `/public/h_minus/` | `MSPRED_ATLAS_DIR/h_minus_out_mgf/`           |
| `/nist23/h_plus/`  | `MSPRED_ATLAS_DIR_NIST/h_plus_out_mgf/`       |
| `/nist23/h_minus/` | `MSPRED_ATLAS_DIR_NIST/h_minus_out_mgf/`      |

---

## Configure NGINX

Copy the NGINX configuration:

```bash
sudo cp deploy/nginx-iceberg-webui.conf \
  /etc/nginx/sites-available/iceberg-webui
```

Enable the site:

```bash
sudo ln -s /etc/nginx/sites-available/iceberg-webui \
  /etc/nginx/sites-enabled/
```

Test and reload:

```bash
sudo nginx -t
sudo systemctl reload nginx
```

---

## Enable HTTPS (Recommended)

Install Certbot:

```bash
sudo apt install certbot python3-certbot-nginx -y
```

Request a certificate:

```bash
sudo certbot --nginx -d your.domain.com
```

---

## Logs and Debugging

### systemd logs

```bash
journalctl -u iceberg-webui -f
```

### NGINX logs

```bash
sudo tail -f /var/log/nginx/error.log
sudo tail -f /var/log/nginx/access.log
```
