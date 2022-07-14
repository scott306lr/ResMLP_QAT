from run_model import save_torchscript_model, load_torchscript_model
quantized_model = load_torchscript_model(model_filepath='qat_weights/epoch2_1.010_78.132_93.728.pth', device='cpu')
quantized_model.eval()

for name, val in quantized_model.state_dict().items():
  print(f"{name}:", end=" ")
  s = val.shape
  print(f"\t{s}")