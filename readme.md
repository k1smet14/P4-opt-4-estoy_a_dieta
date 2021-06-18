# Abstract of Bottom up experiment
- Result of transfer learning

  |pretrained|val f1|
  |---|---|
  |None|0.424|
  |cifar 100 to 10|0.425|
  |cifar 10 to 100|0.433|

- Result of Knowledgee Distillation

  |KD Method| val f1| Augmentation|Optimizer|
  |---|---|---|---|
  |None|0.433|RandAug with cutout|AdamW|
  |KD|0.458|RandAug with cutout|AdamW|
  |SKD|0.453|	RandAug with cutout|AdamW|
  |True Label KD|	0.461|RandAug with cutout|AdamW|
  |True Label KD|0.466|RandAug without cutout|AdamW|
  |KD|0.466|RandAug without cutout|Adam|
  |True Label KD|0.472|RandAug without cutout|Adam|


# NAS
```bash
python tune.py \
--data_config [Your configuration yaml file] \ 
--nas_config [Your configuration yaml file] \
--study-name [Your wandb project name] \
--limit [MACs limit]
```
- data_config : Please refer `config/data/taco.yaml`
- nas_config : Please refer `config/NAS_block/base.yaml`

# From bottomup to KDE
- Step 1
  - train custom model with cifar 10
  ```bash
  python custom_train,py \
  --dataset CIFAR10 \
  --model_path custom_cifar10
  ```

- Step 2
  - transfer learn custom model from cifar10 to cifar100
  ```bash
  python custom_train,py \
  --weight_path custom_cifar10
  --dataset CIFAR100 \
  --model_path custom_cifar100
  ```

- Step 3
  - KDE learning
  ```bash
  python kde_train.py \
  --tweight_path [Folder of your tmodel weight] \
  --sweight_path [Folder of your smodel weight] \
  --model_path [Dst path of your model] \
  --kd_method true_kd
  ```