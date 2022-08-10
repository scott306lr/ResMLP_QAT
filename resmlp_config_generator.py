config_str = ""

for i in range(24):
  layer_str = f"  'layer{i}.quant_act': 8,\n"\
    f"  'layer{i}.norm1': 8,\n"\
    f"  'layer{i}.quant_act1': 8,\n"\
    f"  'layer{i}.attn': 8,\n"\
    f"  'layer{i}.quant_act2': 8,\n"\
    f"  'layer{i}.gamma_1': 8,\n"\
    f"  'layer{i}.quant_act_int32_1': 16,\n"\
    f"  'layer{i}.norm2': 8,\n"\
    f"  'layer{i}.quant_act3': 8,\n"\
    f"  'layer{i}.mlp.fc1': 8,\n"\
    f"  'layer{i}.mlp.quant_act1': 8,\n"\
    f"  'layer{i}.mlp.fc2': 8,\n"\
    f"  'layer{i}.mlp.quant_act2': 8,\n"\
    f"  'layer{i}.gamma_2': 8,\n"\
    f"  'layer{i}.quant_act_int32_2': 16,\n"\
    "\n"\
  
  config_str = config_str + layer_str

head = '"bit_config_resmlp24_uniform8" : {\n'\
  "  'quant_input': 8,\n"\
  "  'quant_patch.proj': 8,\n"\
  "  'quant_act_int32': 16,\n\n"

tail = '},'

print(head + config_str + tail)