import sagemaker
import yaml
from sagemaker.pytorch import PyTorch


if __name__ == "__main__":
    configs_file = "convnet.yaml"

    with open(f"train/configs/{configs_file}") as f:
        configs = yaml.safe_load(f)
        version = configs["fit"]["trainer"]["logger"]["init_args"]["version"].replace(
            ".", "-"
        )

    estimator = PyTorch(
        entry_point="main.py",
        role="arn:aws:iam::757330975535:role/service-role/AmazonSageMaker-ExecutionRole-20231205T113478",
        output_path="s3://db-noise/artifacts",
        code_location="s3://db-noise/artifacts",
        sagemaker_session=sagemaker.Session(default_bucket="db-noise"),
        instance_type="ml.g4dn.xlarge",
        instance_count=1,
        volume_size_in_gb=100,
        max_run=48 * 60 * 60,
        framework_version="2.1",
        py_version="py310",
        source_dir="train",
        hyperparameters={"config": f"configs/{configs_file}"},
        base_job_name=version,
    )
    estimator.fit(
        {
            "train": "s3://db-noise/datasets/train",
            "val": "s3://db-noise/datasets/val",
            "mos": "s3://db-noise/mos",
        }
    )
