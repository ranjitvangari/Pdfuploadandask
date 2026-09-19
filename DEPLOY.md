# Deploying to AWS

This runs the app on a single small EC2 instance: one Docker container for the
Streamlit app, one for [Caddy](https://caddyserver.com/) as a reverse proxy that
automatically gets and renews a free HTTPS certificate for `uploadandask.com`.

Why this instead of ECS/Fargate + a load balancer: for a low-traffic app, the
Application Load Balancer's flat fee (~$16-20/month just for it to exist) costs
more than the entire EC2 box. A single instance is simpler and ~$20/month cheaper.
Revisit this if traffic ever justifies autoscaling.

**Cost estimate:** ~$7-10/month for the instance + ~$1-1.50/month for the domain
and DNS. (AWS App Runner is not an option here — it stopped accepting new
customers on April 30, 2026; AWS now points new users at ECS Express Mode, which
still requires an ALB and costs roughly what's described above for Fargate.)

## Prerequisites

- An AWS account and the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) installed and configured (`aws configure`) with a user/role that can create EC2, Route 53, and IAM resources.
- Your OpenAI API key (already in your local `.env`).

## 1. Register the domain

Register `uploadandask.com` through [Route 53](https://console.aws.amazon.com/route53/home#DomainRegistration:) (~$12/year) or any registrar you prefer. This step needs to happen first since DNS can take a while to propagate, and can run in parallel with the steps below.

If you register elsewhere (Namecheap, GoDaddy, etc.), skip the Route 53 hosted zone in step 4 and instead add the DNS records shown there directly at your registrar.

## 2. Launch the EC2 instance

```bash
# Find the latest Amazon Linux 2023 arm64 AMI (cheap Graviton instances)
AMI_ID=$(aws ssm get-parameter \
  --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64 \
  --query 'Parameter.Value' --output text)

# Create a key pair (skip if you already have one you want to reuse)
aws ec2 create-key-pair --key-name uploadandask-key \
  --query 'KeyMaterial' --output text > uploadandask-key.pem
chmod 400 uploadandask-key.pem

# Security group: SSH from your IP only, HTTP/HTTPS from anywhere
MY_IP=$(curl -s https://checkip.amazonaws.com)
SG_ID=$(aws ec2 create-security-group \
  --group-name uploadandask-sg \
  --description "uploadandask.com web server" \
  --query 'GroupId' --output text)

aws ec2 authorize-security-group-ingress --group-id $SG_ID \
  --protocol tcp --port 22 --cidr ${MY_IP}/32
aws ec2 authorize-security-group-ingress --group-id $SG_ID \
  --protocol tcp --port 80 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id $SG_ID \
  --protocol tcp --port 443 --cidr 0.0.0.0/0

# Launch a t4g.micro (2 vCPU burstable, 1 GB RAM — enough for this app)
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type t4g.micro \
  --key-name uploadandask-key \
  --security-group-ids $SG_ID \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=uploadandask}]' \
  --query 'Instances[0].InstanceId' --output text)

aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Allocate and attach a static IP so it survives reboots
ALLOC_ID=$(aws ec2 allocate-address --query 'AllocationId' --output text)
aws ec2 associate-address --instance-id $INSTANCE_ID --allocation-id $ALLOC_ID
PUBLIC_IP=$(aws ec2 describe-addresses --allocation-ids $ALLOC_ID \
  --query 'Addresses[0].PublicIp' --output text)
echo "Elastic IP: $PUBLIC_IP"
```

## 3. Point the domain at the instance

If using Route 53:

```bash
ZONE_ID=$(aws route53 create-hosted-zone \
  --name uploadandask.com --caller-reference "$(date +%s)" \
  --query 'HostedZone.Id' --output text)

cat > /tmp/dns-record.json <<EOF
{
  "Changes": [
    {"Action": "UPSERT", "ResourceRecordSet": {"Name": "uploadandask.com", "Type": "A", "TTL": 300, "ResourceRecords": [{"Value": "$PUBLIC_IP"}]}},
    {"Action": "UPSERT", "ResourceRecordSet": {"Name": "www.uploadandask.com", "Type": "A", "TTL": 300, "ResourceRecords": [{"Value": "$PUBLIC_IP"}]}}
  ]
}
EOF

aws route53 change-resource-record-sets \
  --hosted-zone-id $ZONE_ID --change-batch file:///tmp/dns-record.json
```

If your domain is registered elsewhere, add these records at your registrar instead:

| Type | Name | Value |
|---|---|---|
| A | `uploadandask.com` (or `@`) | `$PUBLIC_IP` |
| A | `www.uploadandask.com` | `$PUBLIC_IP` |

Wait for DNS to propagate before continuing — check with `dig uploadandask.com`.

## 4. Install Docker and deploy

SSH in and set up the box:

```bash
ssh -i uploadandask-key.pem ec2-user@$PUBLIC_IP

# On the instance:
sudo dnf install -y docker git
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user
exit  # log back in so the group membership takes effect
```

```bash
ssh -i uploadandask-key.pem ec2-user@$PUBLIC_IP

# docker compose plugin
mkdir -p ~/.docker/cli-plugins
curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-aarch64 \
  -o ~/.docker/cli-plugins/docker-compose
chmod +x ~/.docker/cli-plugins/docker-compose

git clone https://github.com/ranjitvangari/Pdfuploadandask.git
cd Pdfuploadandask
cp .env.example .env
nano .env   # paste in your real OPENAI_API_KEY

docker compose up -d --build
```

Caddy will automatically request a Let's Encrypt certificate for `uploadandask.com`
as soon as it can see DNS pointing at this box and reach it on ports 80/443 —
no extra steps needed. Check progress with `docker compose logs caddy -f`.

## 5. Verify

```bash
curl -I https://uploadandask.com
```

Open `https://uploadandask.com` in a browser — you should see the upload/ask UI.

## Updating the app later

```bash
ssh -i uploadandask-key.pem ec2-user@$PUBLIC_IP
cd Pdfuploadandask
git pull
docker compose up -d --build
```

## Security notes

- SSH is restricted to the IP you launched from (`$MY_IP` above). If your IP changes, update the security group rule.
- `.env` holds your real OpenAI key — it's `.gitignore`d and never leaves the instance.
- Consider setting an [AWS Budget alert](https://console.aws.amazon.com/billing/home#/budgets) so an unexpected traffic spike doesn't surprise you on the bill.
