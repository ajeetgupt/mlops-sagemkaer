# Batch Inference - Quick Reference

## 🎯 Overview

The MLOps template now supports **both real-time and batch inference**:

| Mode | Use Case | Trigger | Latency |
|------|----------|---------|---------|
| **Real-time** | Live predictions | API call | <500ms |
| **Batch** | Large datasets | Scheduled/Manual | Minutes-Hours |

---

## 📋 Batch Inference Options

### 1. Manual Batch Job (One-time)

```bash
python scripts/run_batch_inference.py \
  --input-path s3://your-bucket/batch/input/ \
  --output-path s3://your-bucket/batch/output/ \
  --wait
```

**Use when:** Ad-hoc scoring of large datasets

---

### 2. Scheduled Batch Job (Daily/Weekly)

**GitLab CI/CD Schedule:**
1. Go to CI/CD > Schedules
2. Create new schedule: `0 2 * * *` (2 AM daily)
3. Set variable: `SCHEDULE_TYPE=batch`

**Use when:** Regular scoring (daily sales forecasts, weekly churn predictions)

---

### 3. SageMaker Batch Pipeline (Managed)

```bash
# One-time setup
python src/pipelines/batch_pipeline.py --create

# Run pipeline
python src/pipelines/batch_pipeline.py --execute
```

**Use when:** Complex preprocessing + inference + post-processing

---

## 🔧 Configuration

Edit `config/batch_config.yaml`:

```yaml
batch:
  schedule:
    expression: "cron(0 2 * * ? *)"  # Daily at 2 AM
  
  transform:
    instance_type: "ml.m5.xlarge"
    strategy: "MultiRecord"  # Faster for large batches
```

---

## 📊 Monitoring

**CloudWatch Metrics:**
- `ProcessedRecords` - Total records processed
- `FailedRecords` - Failed predictions
- `SuccessRate` - Percentage successful

**Alarms:**
- Batch job failure (>100 failed records)
- Low success rate (<95%)

---

## 🚀 GitLab CI/CD Jobs

| Job | Trigger | Purpose |
|-----|---------|---------|
| `batch-inference-manual` | Manual | One-time batch job |
| `batch-pipeline-create` | Manual | Create SageMaker pipeline |
| `batch-pipeline-execute` | Manual | Run batch pipeline |
| `scheduled-batch-inference` | Schedule | Daily/weekly batch |

---

## 📁 File Structure

```
scripts/
├── run_batch_inference.py      # Main batch script
├── postprocess_batch.py        # Post-processing

src/pipelines/
└── batch_pipeline.py           # SageMaker Pipeline

config/
└── batch_config.yaml           # Batch settings

tests/
└── test_batch.py               # Batch tests
```

---

## 💡 Best Practices

1. **Use MultiRecord strategy** for large datasets (>10K records)
2. **Enable post-processing** for business rules
3. **Monitor success rate** - set alarms for <95%
4. **Partition input data** by date for incremental processing
5. **Use spot instances** for cost savings (add to config)

---

## 🔄 Workflow Example

```mermaid
flowchart LR
    A[Upload to S3] --> B[Trigger Batch Job]
    B --> C[Preprocessing]
    C --> D[Batch Transform]
    D --> E[Post-processing]
    E --> F[Results to S3]
    F --> G[CloudWatch Metrics]
```
