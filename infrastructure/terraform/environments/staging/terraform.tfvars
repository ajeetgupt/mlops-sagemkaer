# Staging Environment Configuration
# ----------------------------------

environment  = "staging"
region       = "us-east-1"
account_id   = "YOUR_ACCOUNT_ID"
team         = "data-science"
project_name = "mlops-sagemaker"

# Networking
vpc_cidr             = "10.1.0.0/16"
private_subnet_cidrs = ["10.1.1.0/24", "10.1.2.0/24"]

# SageMaker (smaller instances for cost)
endpoint_instance_type  = "ml.m5.large"
endpoint_instance_count = 1

# S3
data_bucket_name  = "mlops-data-staging"
model_bucket_name = "mlops-models-staging"
