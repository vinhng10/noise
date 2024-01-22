from sagemaker.processing import FrameworkProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn


if __name__ == "__main__":
    processor = FrameworkProcessor(
        estimator_cls=SKLearn,
        framework_version="0.20.0",
        role="arn:aws:iam::757330975535:role/service-role/AmazonSageMaker-ExecutionRole-20231205T113478",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        volume_size_in_gb=1024,
        max_runtime_in_seconds=24 * 60 * 60,
    )

    processor.run(
        code="process.py",
        source_dir="process",
        inputs=[
            ProcessingInput(
                source="s3://db-noise/headset-training",
                destination="/opt/ml/processing/downloads/headset-training",
            ),
            ProcessingInput(
                source="s3://db-noise/noise-ir",
                destination="/opt/ml/processing/downloads/noise-ir",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="train_data",
                source="/opt/ml/processing/train",
                destination="s3://db-noise/datasets/train",
            )
        ],
    )
