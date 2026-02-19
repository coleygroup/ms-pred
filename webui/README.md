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

Example:

```bash
export FLASK_SECRET_KEY="replace-with-long-random-string"
export MSPRED_ATLAS_DIR="/data/atlas"
export MSPRED_JOB_DIR="/var/lib/iceberg_jobs"
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
Please remember to replace the ``FLASK_SECRET_KEY``!

Reload and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable iceberg-webui
sudo systemctl start iceberg-webui
sudo systemctl status iceberg-webui
```

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
