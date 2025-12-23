import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from pre_sal_ii.improc import colorspace
from pre_sal_ii.improc import scale_image_and_save
from pre_sal_ii.improc import adjust_gamma

import torch.optim as optim
from pre_sal_ii.models.EncoderNN import EncoderNN

def train_porosity_supervised(
        path, path_classes, save_model_path=None, gamma=1.0
    ):
    
    inputImage = cv2.imread(path)
    if gamma != 1.0:
        inputImage = adjust_gamma(inputImage, gamma)

    # BGR to CMKY:
    inputImageCMYK = colorspace.bgr2cmyk(inputImage)

    binaryImage = cv2.inRange(
        inputImageCMYK,
        (92,   0,   0,   0),
        (255, 255,  64, 196))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_ERODE, kernel, iterations=1)
    binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_DILATE, kernel, iterations=1)
    binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_DILATE, kernel, iterations=1)
    binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_ERODE, kernel, iterations=1)

    from skimage.measure import label, regionprops

    label_img = label(binaryImage)
    regions = regionprops(label_img)

    all_objs = []
    for it, region in enumerate(regions):
        ys = (region.coords.T[0] - label_img.shape[0]/2)/(label_img.shape[0]/2)
        xs = (region.coords.T[1] - label_img.shape[1]/2)/(label_img.shape[1]/2)
        obj = {
            "area": region.area,
            "max-dist": max((ys**2 + xs**2)**0.5),
        }
        all_objs.append(obj)

    df = pd.DataFrame(all_objs)

    max_dist = max(df["max-dist"])
    pores_image3 = np.zeros(label_img.shape, dtype=np.uint8)
    for it, region in enumerate(regions):
        if df["max-dist"].iloc[it] <= max_dist*0.8:
            color_value = 255
            pores_image3[region.coords.T[0], region.coords.T[1]] = color_value

    inputImage_cl = cv2.imread(path_classes)
    
    binaryImage_clRed = cv2.inRange(
        inputImage_cl,
        #  B,   G,   R
        (  0,   0, 240),
        (  5,   5, 255))



    ## Extracting features and targets

    from pre_sal_ii.models.WhitePixelRegionDataset import WhitePixelRegionDataset

    from torch.utils.data import DataLoader
    num_samples = 10000
    dataset = WhitePixelRegionDataset(
        pores_image3, inputImage, binaryImage_clRed, num_samples=num_samples)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    # for data in data_loader:
    #     for it, (img, imgTarget) in enumerate(zip(data[0], data[1])):
    #         if it >= 10: break
    #         img = img/255
    #         imgTarget = imgTarget/255
    #         # print(img.numpy().shape, img.dtype)
    #         axes[it//5*2+0, it%5].imshow(img.numpy()[:,:,::-1])
    #         axes[it//5*2+1, it%5].imshow(imgTarget.numpy(), cmap="gray", vmin=0, vmax=1)
    #     break

    model = EncoderNN().to(device)
    # criterion = nn.MSELoss()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=1e-4,
                            # weight_decay=1e-3,
                        )

    best_model_state = None
    best_model_loss = 0

    ## Training

    import copy
    from pre_sal_ii import progress
    import torch.nn.functional as F

    num_epochs = 100
    bar = progress(total=num_epochs*num_samples)
    for epoch in range(num_epochs):
        model.train()
        for data in data_loader:
            imgs = data[0].to(device)
            imgs = imgs.permute(0, 3, 1, 2)
            # print(f"imgs.shape={imgs.shape}")
            imgs = imgs/255
            imgs = F.interpolate(
                imgs, size=(32, 32), mode='bilinear',
                align_corners=False)
            imgs = imgs.reshape(-1, 3*32*32)
            # print(f"imgs.shape={imgs.shape}")
            # break
            outputs = model(imgs)
            expected = data[1].to(device)/255
            expected = torch.squeeze(expected, 2)
            # print(expected)
            # print(f"outputs.shape={outputs.shape}")
            # print(f"expected.shape={expected.shape}")
            # break
            loss = criterion(outputs, expected)
            
            if best_model_state is None or loss.item() < best_model_loss:
                best_model_state = copy.deepcopy(model.state_dict())
                best_model_loss = loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            bar.update(32)

        # print(f"epoch={epoch}, loss={loss.item():.4f}")

    # model = SimpleCNN().to(device)
    # model.load_state_dict(best_model_state)
    # model.eval()

    if save_model_path is not None:
        torch.save(best_model_state, save_model_path)

    model2 = EncoderNN().to(device)
    model2.load_state_dict(best_model_state)
    model2.eval()

    return model2




def apply_model(
        path, model
    ):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if type(model) == str:
        model = EncoderNN().to(device)
        model.load_state_dict(torch.load(model))
        model.eval()

    inputImage = cv2.imread(path)

    # BGR to CMKY:
    inputImageCMYK = colorspace.bgr2cmyk(inputImage)

    binaryImage = cv2.inRange(
        inputImageCMYK,
        (92,   0,   0,   0),
        (255, 255,  64, 196))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_ERODE, kernel, iterations=1)
    binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_DILATE, kernel, iterations=1)
    binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_DILATE, kernel, iterations=1)
    binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_ERODE, kernel, iterations=1)

    from skimage.measure import label, regionprops

    label_img = label(binaryImage)
    regions = regionprops(label_img)

    all_objs = []
    for it, region in enumerate(regions):
        ys = (region.coords.T[0] - label_img.shape[0]/2)/(label_img.shape[0]/2)
        xs = (region.coords.T[1] - label_img.shape[1]/2)/(label_img.shape[1]/2)
        obj = {
            "area": region.area,
            "max-dist": max((ys**2 + xs**2)**0.5),
        }
        all_objs.append(obj)

    df = pd.DataFrame(all_objs)


    max_dist = max(df["max-dist"])
    pores_image3 = np.zeros(label_img.shape, dtype=np.uint8)
    for it, region in enumerate(regions):
        if df["max-dist"].iloc[it] <= max_dist*0.8:
            color_value = 255
            pores_image3[region.coords.T[0], region.coords.T[1]] = color_value

    from pre_sal_ii.models.WhitePixelRegionDataset import WhitePixelRegionDataset

    dataset2 = WhitePixelRegionDataset(
        pores_image3, inputImage, None, num_samples=-1, seed=None)

    pred_image = np.zeros(inputImage.shape, dtype=np.uint8)

    count_gt_half = 0

    with torch.no_grad():
        from pre_sal_ii import progress
        for it, (imgX, _, coords) in enumerate(progress(dataset2)):
            # print(f"coords.shape={coords.shape}")
            imgX = imgX.to(device)
            imgX = imgX.unsqueeze(0)
            imgX = imgX.permute(0, 3, 1, 2)
            # print(f"imgX.shape={imgX.shape}")
            imgX = imgX/255
            imgX = F.interpolate(
                imgX, size=(32, 32), mode='bilinear',
                align_corners=False)
            imgX = imgX.reshape(-1, 3*32*32)
            # print(f"imgX.shape={imgX.shape}")
            # break
            Y = model(imgX)

            pred_image[int(coords[0]), int(coords[1])] = float(Y[0,0])*255

            # if float(Y[0,0]) > 0.5:
            #     count_gt_half += 1
            #     print(f"{[coords[0], coords[1]]} -> {pred_image[coords[0], coords[1]]} (Y[0,0]={Y[0,0]})")
                
            # if it > 1000: break

    return pred_image


if __name__ == "__main__":
    import os
    print(os.getcwd())
    
    os.chdir("../../../notebooks/")
    
    image_name = "ML-tste_original"
    path = f"../data/classificada_01/{image_name}.jpg"
    scale_image_and_save(path, "../out/classificada_01/", 25)
    path = f"../out/classificada_01/{image_name}_25.jpg"

    image_name_classes = "ML-tste_classidicada"
    path_classes = f"../data/classificada_01/{image_name_classes}.jpg"
    scale_image_and_save(path_classes, "../out/classificada_01/", 25)
    path_classes = f"../out/classificada_01/{image_name_classes}_25.jpg"

    image_name_target = "122.20_jpeg_escal"
    path_target = f"../data/thin_sections/{image_name_target}.jpg"
    scale_image_and_save(path_target, "../out/thin_sections_4x/", 25)
    path_target = f"../out/thin_sections_4x/{image_name_target}_25.jpg"
    
    for gamma in np.arange(0.3, 0.71, 0.1):
        os.makedirs("../models/varying_gamma/", exist_ok=True)
        model_path = f"../models/varying_gamma/supervised-1-gamma={gamma:.2f}.bin"
        os.makedirs("../out/varying_gamma_model_preds/", exist_ok=True)
        result_image_path = f"../out/varying_gamma_model_preds/sup_pred_{image_name}_gamma={gamma:.2f}.jpg"
        if not os.path.isfile(result_image_path):
            if os.path.isfile(model_path):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = EncoderNN().to(device)
                model.load_state_dict(torch.load(model_path))
                model.eval()
            else:
                model = train_porosity_supervised(
                    path, path_classes,
                    save_model_path=model_path,
                    gamma=gamma)
            img_result = apply_model(path_target, model)
            cv2.imwrite(result_image_path, img_result)
