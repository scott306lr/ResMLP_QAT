# ResMLP_QAT
Quantization-aware training research for ResMLP model.

```train.py``` to train model with QAT.

```evaluate.py``` to evaluate model accuracy.

put fp32 ***.pth*** file (original  model's weight)  inside ```fp32_weights/``` folder,

put int8 ***.pth*** file (converted model's weight)  inside ```int8_weights/``` folder.

