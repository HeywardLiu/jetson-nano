import timm
import onnx
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "swin_small_patch4_window7_224"
model = timm.create_model("swin_small_patch4_window7_224", pretrained=True, scriptable=True)
model.to(DEVICE).eval()
dummy_input = torch.randn((1, 3, 224, 224)).to(DEVICE)
export_path = "/root/mount-dir/models/onnx-model/{}.onnx".format(MODEL_NAME)

print("Export Model: {}".format(MODEL_NAME))
print("      Device: {}".format(DEVICE))
torch.onnx.export(model=model , args=dummy_input, f=export_path, verbose=True, opset_version=11)
print("\n ----- Export {} to onnx successfully. Saved in: {}".format(MODEL_NAME, export_path))