# Terraform Outputs
# -----------------

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = module.vpc.private_subnet_ids
}

output "sagemaker_execution_role_arn" {
  description = "SageMaker execution role ARN"
  value       = module.security.sagemaker_execution_role_arn
}

output "kms_key_arn" {
  description = "KMS key ARN for encryption"
  value       = module.security.kms_key_arn
}

output "data_bucket_name" {
  description = "S3 bucket for ML data"
  value       = module.s3.data_bucket_name
}

output "model_bucket_name" {
  description = "S3 bucket for model artifacts"
  value       = module.s3.model_bucket_name
}

output "endpoint_name" {
  description = "SageMaker endpoint name"
  value       = module.sagemaker.endpoint_name
}
