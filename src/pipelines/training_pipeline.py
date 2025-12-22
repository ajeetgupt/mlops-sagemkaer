"""
SageMaker Training Pipeline
---------------------------
Defines the complete SageMaker Pipeline for model training, evaluation, and registration.
"""

import json
import logging
import os
from typing import Dict, Optional

import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.xgboost import XGBoost, XGBoostModel

import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_training_pipeline(
    pipeline_name: str = "MLOpsPipeline",
    role: Optional[str] = None,
    s3_bucket: Optional[str] = None,
    region: Optional[str] = None,
    config_path: str = "config/training_config.yaml",
) -> Pipeline:
    """
    Create the SageMaker training pipeline.
    
    Args:
        pipeline_name: Name of the pipeline
        role: SageMaker execution role ARN
        s3_bucket: S3 bucket for artifacts
        region: AWS region
        config_path: Path to training configuration
        
    Returns:
        SageMaker Pipeline object
    """
    # Load configuration
    config = load_config(config_path)
    
    # Initialize session
    session = sagemaker.Session()
    region = region or session.boto_region_name
    role = role or get_execution_role()
    s3_bucket = s3_bucket or session.default_bucket()
    
    logger.info(f"Creating pipeline in region: {region}")
    logger.info(f"Using S3 bucket: {s3_bucket}")
    
    # Define pipeline parameters
    input_data = ParameterString(
        name="InputData",
        default_value=f"s3://{s3_bucket}/{config['training']['input_path']}",
    )
    
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval",
    )
    
    min_auc_threshold = ParameterFloat(
        name="MinAUCThreshold",
        default_value=config["validation"]["min_auc"],
    )
    
    instance_type = ParameterString(
        name="TrainingInstanceType",
        default_value=config["training"]["instance_type"],
    )
    
    instance_count = ParameterInteger(
        name="TrainingInstanceCount",
        default_value=config["training"]["instance_count"],
    )

    # ==========================================
    # Step 1: Data Preprocessing
    # ==========================================
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        base_job_name="mlops-preprocess",
        sagemaker_session=session,
    )
    
    processing_step = ProcessingStep(
        name="PreprocessData",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=input_data,
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output/train",
                destination=f"s3://{s3_bucket}/processed/train",
            ),
            ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/output/validation",
                destination=f"s3://{s3_bucket}/processed/validation",
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/output/test",
                destination=f"s3://{s3_bucket}/processed/test",
            ),
        ],
        code="src/training/preprocess.py",
        job_arguments=[
            "--input-path", "/opt/ml/processing/input",
            "--output-path", "/opt/ml/processing/output",
            "--target-column", config["hyperparameters"].get("target_column", "target"),
        ],
    )

    # ==========================================
    # Step 2: Model Training
    # ==========================================
    hyperparameters = {
        "max_depth": str(config["hyperparameters"]["max_depth"]),
        "eta": str(config["hyperparameters"]["eta"]),
        "gamma": str(config["hyperparameters"]["gamma"]),
        "min_child_weight": str(config["hyperparameters"]["min_child_weight"]),
        "subsample": str(config["hyperparameters"]["subsample"]),
        "colsample_bytree": str(config["hyperparameters"]["colsample_bytree"]),
        "objective": config["hyperparameters"]["objective"],
        "num_round": str(config["hyperparameters"]["num_round"]),
        "eval_metric": config["hyperparameters"]["eval_metric"],
        "early_stopping_rounds": str(config["hyperparameters"]["early_stopping_rounds"]),
    }
    
    xgb_estimator = XGBoost(
        entry_point="train.py",
        source_dir="src/training",
        framework_version="1.7-1",
        hyperparameters=hyperparameters,
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        output_path=f"s3://{s3_bucket}/{config['training']['output_path']}",
        base_job_name="mlops-train",
        sagemaker_session=session,
        enable_sagemaker_metrics=True,
    )
    
    training_step = TrainingStep(
        name="TrainModel",
        estimator=xgb_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )

    # ==========================================
    # Step 3: Model Evaluation
    # ==========================================
    evaluation_processor = ScriptProcessor(
        role=role,
        image_uri=sagemaker.image_uris.retrieve(
            framework="xgboost",
            region=region,
            version="1.7-1",
        ),
        instance_type="ml.m5.large",
        instance_count=1,
        base_job_name="mlops-evaluate",
        sagemaker_session=session,
    )
    
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    
    evaluation_step = ProcessingStep(
        name="EvaluateModel",
        processor=evaluation_processor,
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=processing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{s3_bucket}/evaluation",
            ),
        ],
        code="scripts/evaluate_model.py",
        property_files=[evaluation_report],
    )

    # ==========================================
    # Step 4: Conditional Model Registration
    # ==========================================
    # Condition: Register model only if AUC >= threshold
    condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=evaluation_step.name,
            property_file=evaluation_report,
            json_path="metrics.auc",
        ),
        right=min_auc_threshold,
    )
    
    # Model metrics for registry
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=f"s3://{s3_bucket}/evaluation/evaluation.json",
            content_type="application/json",
        )
    )
    
    # Register model step
    register_step = RegisterModel(
        name="RegisterModel",
        estimator=xgb_estimator,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json", "text/csv"],
        response_types=["application/json", "text/csv"],
        inference_instances=["ml.m5.large", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name="mlops-models",
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )
    
    # Conditional step
    condition_step = ConditionStep(
        name="CheckModelQuality",
        conditions=[condition],
        if_steps=[register_step],
        else_steps=[],  # No action if model doesn't meet threshold
    )

    # ==========================================
    # Create Pipeline
    # ==========================================
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            input_data,
            model_approval_status,
            min_auc_threshold,
            instance_type,
            instance_count,
        ],
        steps=[
            processing_step,
            training_step,
            evaluation_step,
            condition_step,
        ],
        sagemaker_session=session,
    )
    
    return pipeline


def main():
    """Main function to create and optionally execute pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-name", type=str, default="MLOpsPipeline")
    parser.add_argument("--role", type=str, default=None)
    parser.add_argument("--bucket", type=str, default=None)
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--config", type=str, default="config/training_config.yaml")
    parser.add_argument("--create", action="store_true", help="Create/update pipeline")
    parser.add_argument("--execute", action="store_true", help="Execute pipeline")
    parser.add_argument("--local-test", action="store_true", help="Print pipeline definition")
    args = parser.parse_args()
    
    pipeline = create_training_pipeline(
        pipeline_name=args.pipeline_name,
        role=args.role,
        s3_bucket=args.bucket,
        region=args.region,
        config_path=args.config,
    )
    
    if args.local_test:
        # Print pipeline definition for validation
        definition = json.loads(pipeline.definition())
        print(json.dumps(definition, indent=2))
        return
    
    if args.create:
        logger.info("Creating/updating pipeline...")
        pipeline.upsert(role_arn=args.role or get_execution_role())
        logger.info(f"Pipeline '{args.pipeline_name}' created/updated successfully")
    
    if args.execute:
        logger.info("Starting pipeline execution...")
        execution = pipeline.start()
        logger.info(f"Pipeline execution started: {execution.arn}")
        
        # Wait for completion
        execution.wait()
        logger.info(f"Pipeline execution completed with status: {execution.describe()['PipelineExecutionStatus']}")


if __name__ == "__main__":
    main()
