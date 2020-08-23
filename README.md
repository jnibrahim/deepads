# DeepLens Demo

This is a fork from:  
https://bitbucket.org/melbourneITSandbox/deepads/src/master



## Lambda

TO BE ADDED

## Model

### Prepare datasets

The training images needs to be in MXNet RecordIO format before you start
training. Please follow the steps in `deeplens-outwarians-data-preparation.ipynb` or
`deeplens-people-data-preparation.ipynb` to prepare your data. These notebooks
can be executed on Amazon SageMaker Notebook instance.

Note: the steps are identical in both notebook, you can use either of them

### Model Training

Follow the steps in `deeplens-outwarians-training.ipynb` or `deeplens-people-training.ipynb` to train your MXNet image classification model.

Note: the steps are identical in both notebook, you can use either of them

### Model Optimization

MXNet models is better to be optimised for DeepLens' Intel graphics unit, use Intel
deeplearning deployment toolkit on S3 bucket
[here](https://s3.amazonaws.com/deeplens-sagemaker-2bbe16b4-c056-4ae2-9332-d31dd7aeb470/toolkit.tgz)
.

The AWS DeepLens comes with a python module "awscam" to load optimised model.
See DeepLens documentation on
[Device Library](https://docs.aws.amazon.com/deeplens/latest/dg/deeplens-device-library-awscam-model.html)
for more details

There are known limitation for the toolkit, currently it only support limited
type of models for optimization. However you can always use MXNet model directly
in the Lambda function by importing mxnet's python module

A latest version of Intel deeplearning deployment toolkit with modification is installed on our
DeepLens camera at /opt/intel/, you can use this tool to optimise your mxnet model
