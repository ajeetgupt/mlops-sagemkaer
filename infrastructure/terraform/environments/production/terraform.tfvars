# Production Environment Configuration
# ------------------------------------

environment  = "production"
region       = "us-east-1"
account_id   = "YOUR_ACCOUNT_ID"
team         = "data-science"
project_name = "mlops-sagemaker"

# Networking
vpc_cidr             = "10.0.0.0/16"
private_subnet_cidrs = ["10.0.1.0/24", "10.0.2.0/24"]

# SageMaker
endpoint_instance_type  = "ml.m5.xlarge"
endpoint_instance_count = 2

# S3
data_bucket_name  = "mlops-data-production"
model_bucket_name = "mlops-models-production"
