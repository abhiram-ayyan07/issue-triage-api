# AWS Deploy Runbook — GitHub Issue Triage API

Deploys the Dockerized FastAPI service to a single EC2 instance. This is the
simplest possible production path (no ECS/EKS) — good enough for a demo or a
low-traffic personal project, and a reasonable thing to describe in an
interview even if you scale it differently later.

## 1. Provision the EC2 instance

1. Launch an EC2 instance:
   - AMI: Amazon Linux 2023
   - Instance type: `t3.micro` (free-tier eligible; bump to `t3.small` if the
     model or traffic grows)
   - Storage: 20 GB gp3
   - Security group: allow inbound TCP 22 (SSH, restrict to your IP) and TCP
     8000 (or 80 if you put nginx in front) from `0.0.0.0/0`
2. Allocate and associate an Elastic IP so the address doesn't change on
   restart.
3. SSH in: `ssh -i your-key.pem ec2-user@<elastic-ip>`

## 2. Install Docker on the instance

```bash
sudo dnf update -y
sudo dnf install -y docker git
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user
# log out and back in for the group change to take effect
```

## 3. Get the code onto the instance

```bash
git clone <your-repo-url> issue-triage-api
cd issue-triage-api
```

## 4. Build and run

```bash
docker build -t issue-triage-api:latest .
docker run -d \
  --name issue-triage-api \
  --restart unless-stopped \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  issue-triage-api:latest
```

Verify:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "App crashes on startup", "body": "Stack trace attached"}'
```

## 5. Point a domain / add TLS (optional but recommended)

- Point a Route 53 record (or your DNS provider) at the Elastic IP.
- Put nginx or Caddy in front of the container to terminate TLS
  (Let's Encrypt via Certbot, or Caddy's automatic HTTPS), proxying to
  `127.0.0.1:8000`.

## 6. Redeploying after a change

```bash
git pull
docker build -t issue-triage-api:latest .
docker stop issue-triage-api && docker rm issue-triage-api
docker run -d --name issue-triage-api --restart unless-stopped \
  -p 8000:8000 -v $(pwd)/models:/app/models issue-triage-api:latest
```

## 7. Logs and monitoring

```bash
docker logs -f issue-triage-api          # app logs
sqlite3 models/predictions.db "select predicted_label, count(*) from prediction_logs group by 1;"
```

For anything beyond a demo: move `DATABASE_URL` to an RDS Postgres instance,
add CloudWatch agent for metrics/logs, and put the instance behind an
Application Load Balancer with a target group health check on `/health`
instead of exposing the EC2 IP directly.

## Cost note

A `t3.micro` running 24/7 is within (or very close to) the AWS free tier for
the first 12 months of a new account; after that it's on the order of a few
dollars a month. Stop the instance when you're not actively demoing it if
you want to minimize cost during the job search.
