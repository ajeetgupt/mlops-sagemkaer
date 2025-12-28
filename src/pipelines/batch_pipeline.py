"""
Batch Processing Pipeline
-------------------------
SageMaker Pipeline for batch data processing and inference.
"""

import logging
from typing import Dict, Optional

import sagemaker
from sagemaker import get_execution_role
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.transformer import Transformer
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TransformStep

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_batch_pipeline(
    pipeline_name: str = "BatchInferencePipeline",
    role: Optional[str] = None,
    s3_bucket: Optional[str] = None,
    region: Optional[str] = None,
) -> Pipeline:
    """
    Create SageMaker Pipeline for batch inference.
    
    Pipeline steps:
    1. Data preprocessing
    2. Batch transform (inference)
    3. Post-processing (optional)
    
    Args:
        pipeline_name: Name of the pipeline
        role: SageMaker execution role
        s3_bucket: S3 bucket for data
        region: AWS region
        
    Returns:
        SageMaker Pipeline
    """
    session = sagemaker.Session()
    region = region or session.boto_region_name
    role = role or get_execution_role()
    s3_bucket = s3_bucket or session.default_bucket()
    
    logger.info(f"Creating batch pipeline: {pipeline_name}")
    
    # Pipeline parameters
    input_data = ParameterString(
        name="InputData",
        default_value=f"s3://{s3_bucket}/batch/input/",
    )
    
    output_path = ParameterString(
        name="OutputPath",
        default_value=f"s3://{s3_bucket}/batch/output/",
    )
    
    model_name = ParameterString(
        name="ModelName",
        default_value="mlops-model-production",
    )
    
    # Step 1: Preprocessing
    preprocessing_processor = ScriptProcessor(
        role=role,
        image_uri=sagemaker.image_uris.retrieve("sklearn", region, version="1.2-1"),
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name="batch-preprocess",
        sagemaker_session=session,
    )
    
    preprocessing_step = ProcessingStep(
        name="PreprocessBatchData",
        processor=preprocessing_processor,
        inputs=[
            ProcessingInput(
                source=input_data,
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="preprocessed",
                source="/opt/ml/processing/output",
                destination=f"s3://{s3_bucket}/batch/preprocessed/",
            )
        ],
        code="src/training/preprocess.py",
        job_arguments=[
            "--input-path", "/opt/ml/processing/input",
            "--output-path", "/opt/ml/processing/output",
        ],
    )
    
    # Step 2: Batch Transform
    transformer = Transformer(
        model_name=model_name,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        output_path=output_path,
        base_transform_job_name="batch-inference",
        sagemaker_session=session,
        strategy="MultiRecord",
        max_payload=6,
        accept="application/json",
    )
    
    transform_step = TransformStep(
        name="BatchInference",
        transformer=transformer,
        inputs=sagemaker.inputs.TransformInput(
            data=preprocessing_step.properties.ProcessingOutputConfig.Outputs["preprocessed"].S3Output.S3Uri,
            content_type="text/csv",
            split_type="Line",
        ),
    )
    
    # Step 3: Post-processing (optional)
    postprocessing_processor = ScriptProcessor(
        role=role,
        image_uri=sagemaker.image_uris.retrieve("sklearn", region, version="1.2-1"),
        instance_type="ml.m5.large",
        instance_count=1,
        base_job_name="batch-postprocess",
        sagemaker_session=session,
    )
    
    postprocessing_step = ProcessingStep(
        name="PostprocessResults",
        processor=postprocessing_processor,
        inputs=[
            ProcessingInput(
                source=transform_step.properties.TransformOutput.S3OutputPath,
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="final",
                source="/opt/ml/processing/output",
                destination=f"s3://{s3_bucket}/batch/final/",
            )
        ],
        code="scripts/postprocess_batch.py",
        job_arguments=[
            "--input-path", "/opt/ml/processing/input",
            "--output-path", "/opt/ml/processing/output",
        ],
    )
    
    # Create pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[input_data, output_path, model_name],
        steps=[preprocessing_step, transform_step, postprocessing_step],
        sagemaker_session=session,
    )
    
    return pipeline


def main():
    """Main function to create batch pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-name", type=str, default="BatchInferencePipeline")
    parser.add_argument("--role", type=str, default=None)
    parser.add_argument("--bucket", type=str, default=None)
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    
    pipeline = create_batch_pipeline(
        pipeline_name=args.pipeline_name,
        role=args.role,
        s3_bucket=args.bucket,
    )
    
    if args.create:
        logger.info("Creating/updating batch pipeline...")
        pipeline.upsert(role_arn=args.role or get_execution_role())
        logger.info(f"Pipeline '{args.pipeline_name}' created successfully")
    
    if args.execute:
        logger.info("Starting batch pipeline execution...")
        execution = pipeline.start()
        logger.info(f"Execution started: {execution.arn}")


if __name__ == "__main__":
    main()
