import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import v2
from torchvision.utils import make_grid

from improved.data_transforms import prepreprocess
from improved.eval import predict
from improved.model import AlexNetImproved

classes = [
    "actinic keratosis",
    "basal cell carcinoma",
    "dermatofibroma",
    "healthy",
    "melanoma",
    "nevus",
    "pigmented benign keratosis",
    "seborrheic keratosis",
    "squamous cell carcinoma",
    "vascular lesion",
]

model: AlexNetImproved = AlexNetImproved(num_classes=len(classes))

# model_name = "skin.png"
# model.load_state_dict(
#     torch.load("models/skin.model.pt", map_location=torch.device("cpu"))
# )
# preprocess = torch.load("models/skin.preprocess.pt", map_location=torch.device("cpu"))

model_name = "imagenette.png"
model.load_state_dict(
    torch.load("models/improved_imagenette.model.pt", map_location=torch.device("cpu"))
)
preprocess = torch.load(
    "models/improved_imagenette.preprocess.pt", map_location=torch.device("cpu")
)

# conv1 = model.network[0][0].weight.data.numpy()
# conv1 = np.transpose(conv1, [0, 2, 3, 1])
# print(conv1.shape)
# print(conv1[0])
# print(conv1.shape)
# print(conv1[0])

# transforms = v2.Compose(
#     [
#         v2.ToDtype(torch.uint8, scale=True),
#         # v2.ToPILImage(),
#     ]
# )
# print(transforms(conv1[0]))

kernels = model.network[0][0].weight.detach().clone()
print(type(kernels))
kernels = kernels - kernels.min()
kernels = kernels / kernels.max()
img = make_grid(kernels, nrow=16)
plt.imshow(img.permute(1, 2, 0), interpolation="nearest")
plt.savefig(model_name, dpi=1200)
plt.show()
