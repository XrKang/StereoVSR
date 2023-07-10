import torch

checkpoint = torch.load(val_path+args.resume)
state_dict = net.state_dict()
torch.save({'state_dict':state_dict},val_path+args.resume,_use_new_zipfile_serialization=False)