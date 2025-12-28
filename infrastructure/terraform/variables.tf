# Variables for MLOps Infrastructure
# -----------------------------------

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
}

variable "account_id" {
  description = "AWS account ID"
  type        = string
}

variable "team" {
  description = "Team name for tagging"
  type        = string
  default     = "data-science"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "mlops-sagemaker"
}

# Networking
variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnet_cidrs" {
  description = "Private subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

# SageMaker
variable "endpoint_instance_type" {
  description = "SageMaker endpoint instance type"
  type        = string
  default     = "ml.m5.large"
}

variable "endpoint_instance_count" {
  description = "Number of endpoint instances"
  type        = number
  default     = 1
}

# S3
variable "data_bucket_name" {
  description = "S3 bucket for ML data"
  type        = string
}

variable "model_bucket_name" {
  description = "S3 bucket for model artifacts"
  type        = string
}
