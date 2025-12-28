# Security Module
# ---------------
# IAM roles, KMS keys, and secrets management

# KMS Key for encryption
resource "aws_kms_key" "mlops" {
  description             = "KMS key for MLOps encryption"
  deletion_window_in_days = 7
  enable_key_rotation     = true
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${var.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow SageMaker Service"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:GenerateDataKey*"
        ]
        Resource = "*"
      }
    ]
  })
  
  tags = {
    Name = "${var.project_name}-kms-${var.environment}"
  }
}

resource "aws_kms_alias" "mlops" {
  name          = "alias/${var.project_name}-${var.environment}"
  target_key_id = aws_kms_key.mlops.key_id
}

# SageMaker Execution Role (Least Privilege)
resource "aws_iam_role" "sagemaker_execution" {
  name = "${var.project_name}-sagemaker-execution-${var.environment}"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

# S3 Access Policy (Specific Buckets Only)
resource "aws_iam_role_policy" "sagemaker_s3" {
  name = "sagemaker-s3-access"
  role = aws_iam_role.sagemaker_execution.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          var.data_bucket_arn,
          "${var.data_bucket_arn}/*",
          var.model_bucket_arn,
          "${var.model_bucket_arn}/*"
        ]
      }
    ]
  })
}

# CloudWatch Logs Policy
resource "aws_iam_role_policy" "sagemaker_logs" {
  name = "sagemaker-logs-access"
  role = aws_iam_role.sagemaker_execution.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:${var.account_id}:log-group:/aws/sagemaker/*"
      }
    ]
  })
}

# ECR Access Policy
resource "aws_iam_role_policy" "sagemaker_ecr" {
  name = "sagemaker-ecr-access"
  role = aws_iam_role.sagemaker_execution.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:BatchCheckLayerAvailability"
        ]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = "ecr:GetAuthorizationToken"
        Resource = "*"
      }
    ]
  })
}

# KMS Access Policy
resource "aws_iam_role_policy" "sagemaker_kms" {
  name = "sagemaker-kms-access"
  role = aws_iam_role.sagemaker_execution.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:GenerateDataKey*"
        ]
        Resource = aws_kms_key.mlops.arn
      }
    ]
  })
}

# SNS Topic for Alerts
resource "aws_sns_topic" "mlops_alerts" {
  name              = "${var.project_name}-alerts-${var.environment}"
  kms_master_key_id = aws_kms_key.mlops.id
}

# Variables
variable "environment" {
  type = string
}

variable "project_name" {
  type = string
}

variable "account_id" {
  type = string
}

variable "data_bucket_arn" {
  type = string
}

variable "model_bucket_arn" {
  type = string
}

# Outputs
output "sagemaker_execution_role_arn" {
  value = aws_iam_role.sagemaker_execution.arn
}

output "kms_key_arn" {
  value = aws_kms_key.mlops.arn
}

output "sns_topic_arn" {
  value = aws_sns_topic.mlops_alerts.arn
}
