import torch
import torchvision
import timm
import argparse
from torch.profiler import profile, record_function, ProfilerActivity

def prof_model(pt_model: torch.nn.Module, device="cuda"):
    print("Profiling... Device: {}" .format(device))
    pt_model.to(device).eval()
    dummy_input = torch.randn((1, 3, 224, 224)).to(device)
    activities=[ProfilerActivity.CPU]
    if device=="cuda":
        activities.append(ProfilerActivity.CUDA)
        
    with profile(activities=activities,
                 with_stack=True, with_flops=True, with_modules=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            pt_model(dummy_input)
            
    print(prof.key_averages().table(sort_by="self_{}_time_total".format(device)))
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
  
if __name__=="__main__":
    MODEL_NAME = "deit_small_distilled_patch16_224"
    DEVICE = "cpu"

    model = timm.models.create_model(MODEL_NAME, pretrained=True, exportable=True)
    model.to(DEVICE).eval()
    prof_model(pt_model=model, device=DEVICE)