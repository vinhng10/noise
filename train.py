import sagemaker
from sagemaker.pytorch import PyTorch


if __name__ == "__main__":
    sess = sagemaker.Session(default_bucket="db-noise")
    # role = sess.get_caller_identity_arn()
    # inputs = sess.upload_data(
    #     path="data", bucket=sess.default_bucket(), key_prefix="train"
    # )
    # print(role)

    estimator = PyTorch(
        entry_point="main.py",
        role="arn:aws:iam::757330975535:role/service-role/AmazonSageMaker-ExecutionRole-20231205T113478",
        output_path="s3://db-noise/artifacts",
        code_location="s3://db-noise/artifacts",
        sagemaker_session=sess,
        instance_type="ml.g4dn.xlarge",
        instance_count=1,
        volume_size_in_gb=100,
        framework_version="2.1",
        py_version="py310",
        source_dir="train",
        hyperparameters={},
    )
    estimator.fit({"train": "s3://db-noise/datasets/train"})
