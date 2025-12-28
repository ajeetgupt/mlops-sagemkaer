# SageMaker Module
# ----------------
# Model package group and endpoint configuration

# Model Package Group (Registry)
resource "aws_sagemaker_model_package_group" "main" {
  model_package_group_name        = "${var.project_name}-models-${var.environment}"
  model_package_group_description = "Model registry for ${var.project_name}"
  
  tags = {
    Name = "${var.project_name}-models-${var.environment}"
  }
}

# Domain for SageMaker Studio (optional)
resource "aws_sagemaker_domain" "main" {
  domain_name = "${var.project_name}-studio-${var.environment}"
  auth_mode   = "IAM"
  vpc_id      = var.vpc_id
  subnet_ids  = var.subnet_ids
  
  default_user_settings {
    execution_role  = var.sagemaker_role_arn
    security_groups = [var.security_group_id]
  }
  
  retention_policy {
    home_efs_file_system = "Delete"
  }
  
  tags = {
    Name = "${var.project_name}-studio-${var.environment}"
  }
}

# Variables
variable "environment" {
  type = string
}

variable "project_name" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "subnet_ids" {
  type = list(string)
}

variable "security_group_id" {
  type = string
}

variable "sagemaker_role_arn" {
  type = string
}

variable "kms_key_arn" {
  type = string
}

variable "model_bucket_name" {
  type = string
}

# Outputs
output "model_package_group_name" {
  value = aws_sagemaker_model_package_group.main.model_package_group_name
}

output "endpoint_name" {
  value = "${var.project_name}-endpoint-${var.environment}"
}
