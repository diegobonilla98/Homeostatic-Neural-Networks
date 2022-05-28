import matplotlib.pyplot as plt
import torch
from DataLoaderDogCat import DogCats
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd


model_path = './checkpoints/homeostatic_best_model.pth'
model = torch.load(model_path)
model = model.cuda()
model.eval()

class_names = ['dog', 'cat']
data_loader = DogCats((224, 224), augment=True, dataset_path='./validation_data_cutre/*')

correct = 0
history = []

with torch.no_grad():
    for sample in data_loader:
        s_image, s_label, original = sample['image'], sample['class'], sample['original']
        s_label = s_label.long()

        s_image = s_image.cuda()
        s_label = s_label.cuda()

        s_image = s_image.unsqueeze(0)

        s_class_output = model(s_image)
        s_class_output_probs = torch.exp(s_class_output)
        label_y = np.argmax(s_class_output_probs[0].cpu().data.numpy())

        correct_label = s_label.cpu().data.numpy().flatten()[0]
        correct += label_y == correct_label

        history.append([label_y, correct_label])

        # plt.title(f'Predicted: {class_names[label_y]}')
        # plt.imshow(original)
        # plt.show()


history = np.array(history)
cm = confusion_matrix(history[:, 1], history[:, 0])


plt.title("Confusion Matrix on Test Data")
df_cm = pd.DataFrame(cm, range(2), range(2))
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})

plt.show()
