# Terraform Backend Configuration
# --------------------------------
# Remote state storage in S3 with DynamoDB locking

terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    bucket         = "mlops-terraform-state-${var.account_id}"
    key            = "mlops/sagemaker/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "mlops-terraform-locks"
  }
}

provider "aws" {
  region = var.region
  
  default_tags {
    tags = {
      Project     = "mlops-sagemaker"
      Environment = var.environment
      ManagedBy   = "terraform"
      Team        = var.team
    }
  }
}
