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
        instance_type="ml.g4dn.xlarge",
        instance_count=1,
        framework_version="2.1",
        py_version="py310",
        source_dir="core",
        hyperparameters={},
    )
    estimator.fit({"train": "s3://db-noise/train"})
