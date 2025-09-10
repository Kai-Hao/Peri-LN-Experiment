import matplotlib.pyplot as plt
import numpy as np
import json
import os

folder_names = [
    'resnet50-PATCH_0_conlrmin_False_False_600_Adan',
    'resnet50-PATCH_0_conlrmin_False_True_600_Adan',
    'resnet50-PATCH_0_conlrmin_True_False_600_Adan',
    'resnet50-PATCH_0_conlrmin_True_True_600_Adan',
]

labels = [
    'Optimizer:Adan Pre-LN:False Post-LN:False',
    'Optimizer:Adan Pre-LN:False Post-LN:True',
    'Optimizer:Adan Pre-LN:True  Post-LN:False',
    'Optimizer:Adan Pre-LN:True  Post-LN:True',
]
plt.figure(figsize = (8, 6))
for idx, folder_name in enumerate(folder_names):
    records_path = os.path.join('./records', folder_name, 'records.json')
    with open(records_path, 'r') as f:
        records = json.load(f)
    plt.plot(records['TEST_ACCS'], label = f'{labels[idx]}')

plt.title('Training Traversal on CUB 200 2011 Test Datasets', fontsize = 20, pad = 10)
plt.xlim(0, 70)
# plt.ylim(0, 100)
plt.ylabel('Accuracy', fontsize = 20)
plt.xlabel('Epochs', fontsize = 20)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.grid('--', alpha = 0.4)
plt.legend(fontsize = 12)
plt.savefig('training_traversal_cub2002011_testacc.png', dpi = 300)

plt.cla()
plt.clf()
plt.close()

plt.figure(figsize = (8, 6))
for idx, folder_name in enumerate(folder_names):
    records_path = os.path.join('./records', folder_name, 'records.json')
    with open(records_path, 'r') as f:
        records = json.load(f)
    plt.plot(records['TEST_LOSSES'], label = f'{labels[idx]}')

plt.title('Training Traversal on CUB 200 2011 Test Datasets', fontsize = 20, pad = 10)
plt.xlim(0, 70)
# plt.ylim(0, 100)
plt.ylabel('Loss', fontsize = 20)
plt.xlabel('Epochs', fontsize = 20)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.grid('--', alpha = 0.4)
plt.legend(fontsize = 12)
plt.savefig('training_traversal_cub2002011_testloss.png', dpi = 300)

plt.cla()
plt.clf()
plt.close()

plt.figure(figsize = (8, 6))
for idx, folder_name in enumerate(folder_names):
    records_path = os.path.join('./records', folder_name, 'records.json')
    with open(records_path, 'r') as f:
        records = json.load(f)
    plt.plot(records['TRAIN_ACCS'], label = f'{labels[idx]}')

plt.title('Training Traversal on CUB 200 2011 Train Datasets', fontsize = 20, pad = 10)
plt.xlim(0, 70)
# plt.ylim(0, 100)
plt.ylabel('Accuracy', fontsize = 20)
plt.xlabel('Epochs', fontsize = 20)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.grid('--', alpha = 0.4)
plt.legend(fontsize = 12)
plt.savefig('training_traversal_cub2002011_trainacc.png', dpi = 300)

plt.cla()
plt.clf()
plt.close()

plt.figure(figsize = (8, 6))
for idx, folder_name in enumerate(folder_names):
    records_path = os.path.join('./records', folder_name, 'records.json')
    with open(records_path, 'r') as f:
        records = json.load(f)
    plt.plot(records['TRAIN_LOSSES'], label = f'{labels[idx]}')

plt.title('Training Traversal on CUB 200 2011 Train Datasets', fontsize = 20, pad = 10)
plt.xlim(0, 70)
# plt.ylim(0, 100)
plt.ylabel('Loss', fontsize = 20)
plt.xlabel('Epochs', fontsize = 20)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.grid('--', alpha = 0.4)
plt.legend(fontsize = 12)
plt.savefig('training_traversal_cub2002011_trainloss.png', dpi = 300)

plt.cla()
plt.clf()
plt.close()