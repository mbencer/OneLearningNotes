import subprocess
import time
import os
import shutil

install_dir = '/home/m.bencer/workspace/one_install'
onert_train_path = f'{install_dir}/bin/onert_train'
script_dir = os.path.dirname(os.path.realpath(__file__))
export_model_path = f'{script_dir}/trained_model.circle'
onert_defalt_args = f'--loss 1 --loss_reduction_type 1 --learning_rate 0.00001 --batch_size 32'
comparator_path = f"{script_dir}/circle_compare_weights/build/compare_weights"

class TestConfig:
    def __init(self):
        self.model_file = ''
        self.model_path = ''
        self.epoch_number = []
        self.batches_in_epoch = []
        self.frozen_ops_idx = []
        self.optimizer = '1'
        self.gen_train_data_method = None

test_configs = []
mobilenetv2 = TestConfig()
mobilenetv2.model_file = "customized_mobilenetv2.circle"
mobilenetv2.model_path = f'{script_dir}/test_models/mobilenetv2/{mobilenetv2.model_file}'
mobilenetv2.epoch_number = [1, 5, 10]
mobilenetv2.batches_in_epoch = [10, 30, 50]
# mobilenetv2.epoch_number = [1, 2]
# mobilenetv2.batches_in_epoch = [10, 30]
mobilenetv2.frozen_ops_idx = ['', '0-68', '0-63', '0-30', '0-10']
mobilenetv2.optimizer = '1'
from training_data.mobilenet_data.gen_mobilenet_train_data import generate_train_data as mobilenet_gen
mobilenetv2.gen_train_data_method = mobilenet_gen
test_configs.append(mobilenetv2)

conv_mnist = TestConfig()
conv_mnist.model_file = "custom_conv_mnist_model.circle"
conv_mnist.model_path = f'{script_dir}/test_models/conv_mnitst_model/{conv_mnist.model_file}'
conv_mnist.epoch_number = [10, 20, 30]
conv_mnist.batches_in_epoch = [100, 200, 300]
# conv_mnist.epoch_number = [5, 10]
# conv_mnist.batches_in_epoch = [20, 30]
conv_mnist.frozen_ops_idx = ['', '0-6', '0-5', '0-1']
conv_mnist.optimizer = '2'
from training_data.mnist_data.gen_mnist_train_data import generate_train_data as mnist_gen
conv_mnist.gen_train_data_method = mnist_gen
test_configs.append(conv_mnist)

def find_changed_weights(compare_res_str):
    result = {}
    for line in compare_res_str.split('\n'):
        if line:
            idx = int(line.split()[0])
            changed = line.split()[-1] == 'changed'
            result[idx] = changed
    return result

def extract_train_result(train_result):
    result = ''
    for line in train_result.split('\n'):
        if line.startswith('Epoch ') or line.startswith('EXECUTE'):
            result += '- ' + line + '<br>'
            
    return result

def check_if_frozen_weights_not_changed(frozen_ops_idx, changed_weights_idx):
    if len(changed_weights_idx) > 0:
        expected_not_changed_idx = []
        for idx_str in frozen_ops_idx.split(','):
            if '-' in idx_str:
                start, end = idx_str.split('-')
                expected_not_changed_idx += list(range(int(start), int(end)+1))
            else:
                expected_not_changed_idx.append(int(idx_str))
        for idx in expected_not_changed_idx:
            if idx in changed_weights_idx:
                if changed_weights_idx[idx]: # a frozen weight was changed
                    return False
    return True

def run_command(command):
    start_time = time.time()
    result = subprocess.run(command, stdout = subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, check=True, universal_newlines = True)
    end_time = time.time()
    return result.stdout, end_time - start_time

report = '| test model | number of epochs | batches in epoch | frozen nodes idx | result of weights comparison | Accuracy/Time |\n'
report += '| ---------- | ---------------- | ---------------- | ---------------- | ---------------------------- | ------------- |\n'
for test_config in test_configs:
    for epoch_number in test_config.epoch_number:
        for batches_in_epoch in test_config.batches_in_epoch:
            in_data_path, out_data_path = test_config.gen_train_data_method(batches_in_epoch)
            for frozen_ops_idx in test_config.frozen_ops_idx:
                frozen_ops_idx_provided = len(frozen_ops_idx) > 0
                train_command = f"{onert_train_path} {test_config.model_path} --epoch {epoch_number} {onert_defalt_args} --load_expected:raw {out_data_path} --load_input:raw {in_data_path} --optimizer {test_config.optimizer}"
                if frozen_ops_idx_provided:
                    train_command += f' --frozen_ops_idx {frozen_ops_idx}'
                    train_command += f' --export_path {export_model_path}'
                    shutil.copyfile(test_config.model_path, export_model_path)
                train_cmd_res,_ = run_command(train_command)
                comparision_result = 'NA'
                if frozen_ops_idx_provided:
                    compare_command = f'{comparator_path} {test_config.model_path} {export_model_path}'
                    compare_res_str,_ = run_command(compare_command)
                    os.remove(export_model_path)
                    changed_weights_idx = find_changed_weights(compare_res_str)
                    comparision_result = 'PASS' if check_if_frozen_weights_not_changed(frozen_ops_idx, changed_weights_idx) else 'FAIL'
                result = f'| {test_config.model_file} | {epoch_number} | {batches_in_epoch} | {frozen_ops_idx} | {comparision_result} | {extract_train_result(train_cmd_res)} |'
                report += result + '\n'

with open("report.md", "w") as report_file:
    report_file.write(report)
