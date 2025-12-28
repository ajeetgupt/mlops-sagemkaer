# Networking Module
# -----------------
# VPC, subnets, and security groups for SageMaker isolation

# VPC
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "${var.project_name}-vpc-${var.environment}"
  }
}

# Private Subnets (for SageMaker)
resource "aws_subnet" "private" {
  count             = length(var.private_subnet_cidrs)
  vpc_id            = aws_vpc.main.id
  cidr_block        = var.private_subnet_cidrs[count.index]
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  tags = {
    Name = "${var.project_name}-private-${count.index + 1}-${var.environment}"
    Type = "private"
  }
}

data "aws_availability_zones" "available" {
  state = "available"
}

# Security Group for SageMaker
resource "aws_security_group" "sagemaker" {
  name_prefix = "${var.project_name}-sagemaker-"
  vpc_id      = aws_vpc.main.id
  
  # Inbound: Only from VPC
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
    description = "HTTPS from VPC"
  }
  
  # Outbound: S3 and SageMaker services via VPC endpoints
  egress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTPS to AWS services"
  }
  
  tags = {
    Name = "${var.project_name}-sagemaker-sg-${var.environment}"
  }
  
  lifecycle {
    create_before_destroy = true
  }
}

# VPC Endpoints for private connectivity
resource "aws_vpc_endpoint" "s3" {
  vpc_id            = aws_vpc.main.id
  service_name      = "com.amazonaws.${data.aws_region.current.name}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = [aws_route_table.private.id]
  
  tags = {
    Name = "${var.project_name}-s3-endpoint-${var.environment}"
  }
}

resource "aws_vpc_endpoint" "sagemaker_api" {
  vpc_id              = aws_vpc.main.id
  service_name        = "com.amazonaws.${data.aws_region.current.name}.sagemaker.api"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = aws_subnet.private[*].id
  security_group_ids  = [aws_security_group.sagemaker.id]
  private_dns_enabled = true
  
  tags = {
    Name = "${var.project_name}-sagemaker-api-endpoint-${var.environment}"
  }
}

resource "aws_vpc_endpoint" "sagemaker_runtime" {
  vpc_id              = aws_vpc.main.id
  service_name        = "com.amazonaws.${data.aws_region.current.name}.sagemaker.runtime"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = aws_subnet.private[*].id
  security_group_ids  = [aws_security_group.sagemaker.id]
  private_dns_enabled = true
  
  tags = {
    Name = "${var.project_name}-sagemaker-runtime-endpoint-${var.environment}"
  }
}

# Route Table for private subnets
resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id
  
  tags = {
    Name = "${var.project_name}-private-rt-${var.environment}"
  }
}

resource "aws_route_table_association" "private" {
  count          = length(aws_subnet.private)
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

data "aws_region" "current" {}

# Variables
variable "vpc_cidr" {
  type = string
}

variable "private_subnet_cidrs" {
  type = list(string)
}

variable "environment" {
  type = string
}

variable "project_name" {
  type = string
}

# Outputs
output "vpc_id" {
  value = aws_vpc.main.id
}

output "private_subnet_ids" {
  value = aws_subnet.private[*].id
}

output "sagemaker_security_group_id" {
  value = aws_security_group.sagemaker.id
}
