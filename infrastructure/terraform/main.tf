# Main Terraform Configuration
# ----------------------------
# Provisions SageMaker infrastructure with security best practices

# ========================================
# VPC and Networking (Private)
# ========================================
module "vpc" {
  source = "./modules/networking"
  
  vpc_cidr             = var.vpc_cidr
  private_subnet_cidrs = var.private_subnet_cidrs
  environment          = var.environment
  project_name         = var.project_name
}

# ========================================
# S3 Buckets with Encryption
# ========================================
module "s3" {
  source = "./modules/s3"
  
  data_bucket_name  = var.data_bucket_name
  model_bucket_name = var.model_bucket_name
  environment       = var.environment
  kms_key_arn       = module.security.kms_key_arn
}

# ========================================
# Security (IAM, KMS, Secrets)
# ========================================
module "security" {
  source = "./modules/security"
  
  environment  = var.environment
  project_name = var.project_name
  account_id   = var.account_id
  
  # S3 buckets for policy
  data_bucket_arn  = module.s3.data_bucket_arn
  model_bucket_arn = module.s3.model_bucket_arn
}

# ========================================
# SageMaker
# ========================================
module "sagemaker" {
  source = "./modules/sagemaker"
  
  environment           = var.environment
  project_name          = var.project_name
  vpc_id                = module.vpc.vpc_id
  subnet_ids            = module.vpc.private_subnet_ids
  security_group_id     = module.vpc.sagemaker_security_group_id
  sagemaker_role_arn    = module.security.sagemaker_execution_role_arn
  kms_key_arn           = module.security.kms_key_arn
  model_bucket_name     = var.model_bucket_name
}

# ========================================
# Monitoring
# ========================================
module "monitoring" {
  source = "./modules/monitoring"
  
  environment    = var.environment
  project_name   = var.project_name
  endpoint_name  = module.sagemaker.endpoint_name
  sns_topic_arn  = module.security.sns_topic_arn
}
