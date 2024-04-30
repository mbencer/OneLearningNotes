# Fine tuning

The fine tuning feature was implemented in [PR](todo)

## How to use
In order to disable traning of some nodes in the model, the new `--frozen_ops_idx` flag of `onert_train` executable should be used. The indexes can be passed as a comma-separated list (like 65,68,70) or in range form (like 60-70). Note that the numeration starts from 0 and trainable and untrainable nodes can be mixed (like `--frozen_ops_idx 20-30, 51,60-70`).
Example command:
```
~/workspace/one_install/bin/onert_train test_models/conv_mnitst_model/custom_conv_mnist_model.circle --epoch 30 --load_expected:raw ./training_data/mnist_data/mnist.output.bin --load_input:raw ./training_data/mnist_data/mnist.input.bin --optimizer 2 --loss 1 --loss_reduction_type 1 --learning_rate 0.00001 --batch_size 32 --frozen_ops_idx 0-6
```

## Verification
### Test models
1. Customized mobilenetv2 model
- [customized_mobilenetv2.tflite](./test_models/mobilenetv2/customized_mobilenetv2.tflite) created via [generate_ustomized_mobilenetv2.py](./test_models/mobilenetv2/generate_customized_mobilenetv2.py)
- [customized_mobilenetv2.tflite](./test_models/mobilenetv2/customized_mobilenetv2.tflite) was converted to [customized_mobilenetv2.circle](./test_models/mobilenetv2/customized_mobilenetv2.circle) via command:
```
~/workspace/one_install/bin/one-prepare-venv
~/workspace/one_install/bin/one-import tflite -i test_models/mobilenetv2/customized_mobilenetv2.tflite -o test_models/mobilenetv2/customized_mobilenetv2.circle
```
- training data data [cats_and_dogs.input.bin](./training_data/mobilenet_data/cats_and_dogs.input.bin) and [cats_and_dogs.output.bin](./training_data/mobilenet_data/cats_and_dogs.output.bin) were generated via [gen_mobilenet_train_data.py](./training_data/mobilenet_data/gen_mobilenet_train_data.py) script

2. Custom convolution model
- [custom_conv_mnist_model.tflite](./test_models/conv_mnitst_model/custom_conv_mnist_model.tflite) was created via [generate_customized_conv_model.py](./test_models/conv_mnitst_model/generate_customized_conv_model.py)
- [custom_conv_mnist_model.tflite](./test_models/conv_mnitst_model/custom_conv_mnist_model.tflite) was converted to [custom_conv_mnist_model.circle](./test_models/conv_mnitst_model/custom_conv_mnist_model.circle) via command:
```
~/workspace/one_install/bin/one-import tflite -i test_models/conv_mnitst_model/custom_conv_mnist_model.tflite -o test_models/conv_mnitst_model/custom_conv_mnist_model.circle
```

### Results
PRE: 
1. Build ONE project with `-DBUILD_ONERT_TRAIN=ON -DCMAKE_INSTALL_PREFIX="~/workspace/one_install"`
2. Build helper C++ program to compare weights of model before and after traning.
```
cd circle_compare_weights
mkdir build
cmake -DCMAKE_BUILD_TYPE=Debug -DONE_INSTALL_DIR="~/workspace/one_install" -DCIRCLE_SCHEMA_DIR="~/workspace/ONE/runtime/libs/circle-schema" ..
make -j $(nproc)
```
where `~/workspace/ONE` is ONE repo path.

The report with results was generated via [report.py](./report.py) script. Comparison of weights before and after fine-tuning was done via [helper progran](./circle_compare_weights/main.cpp) script used underneath.
Many combinations of bath size, data size, epoch and frozen ops is tested. The script test if frozen nodes weights are not changed and collect time and accuracy of traning.

[RESULT REPORT](report.md)
