# Monitoring Module
# -----------------
# CloudWatch alarms and dashboard

# Endpoint Latency Alarm
resource "aws_cloudwatch_metric_alarm" "endpoint_latency" {
  alarm_name          = "${var.project_name}-high-latency-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "ModelLatency"
  namespace           = "AWS/SageMaker"
  period              = 60
  statistic           = "p95"
  threshold           = 500
  alarm_description   = "Endpoint latency exceeds 500ms"
  alarm_actions       = [var.sns_topic_arn]
  
  dimensions = {
    EndpointName = var.endpoint_name
    VariantName  = "AllTraffic"
  }
  
  tags = {
    Name = "${var.project_name}-latency-alarm-${var.environment}"
  }
}

# Error Rate Alarm
resource "aws_cloudwatch_metric_alarm" "endpoint_errors" {
  alarm_name          = "${var.project_name}-high-errors-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "Invocation5XXErrors"
  namespace           = "AWS/SageMaker"
  period              = 60
  statistic           = "Sum"
  threshold           = 10
  alarm_description   = "Endpoint error rate too high"
  alarm_actions       = [var.sns_topic_arn]
  
  dimensions = {
    EndpointName = var.endpoint_name
    VariantName  = "AllTraffic"
  }
  
  tags = {
    Name = "${var.project_name}-errors-alarm-${var.environment}"
  }
}

# Dashboard
resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "${var.project_name}-${var.environment}"
  
  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          title   = "Invocations"
          metrics = [["AWS/SageMaker", "Invocations", "EndpointName", var.endpoint_name]]
          period  = 60
          stat    = "Sum"
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        properties = {
          title = "Latency (p50, p95, p99)"
          metrics = [
            ["AWS/SageMaker", "ModelLatency", "EndpointName", var.endpoint_name, { stat = "p50" }],
            ["...", { stat = "p95" }],
            ["...", { stat = "p99" }]
          ]
          period = 60
        }
      }
    ]
  })
}

# Variables
variable "environment" {
  type = string
}

variable "project_name" {
  type = string
}

variable "endpoint_name" {
  type = string
}

variable "sns_topic_arn" {
  type = string
}
