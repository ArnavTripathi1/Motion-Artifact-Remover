import numpy as np
import torch
import torchvision
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
from PIL import Image
from tqdm.auto import tqdm
from timeit import default_timer as Timer

def print_train_time(start,
                     end,
                     device):
	total_time = end - start
	print(f"\nTrain time on {device}: {total_time:.3f} seconds")

def compute_psnr_ssim(tensor1, tensor2):
    psnr_list = []
    ssim_list = []

    tensor1 = tensor1.detach().cpu().numpy()
    tensor2 = tensor2.detach().cpu().numpy()

    for i in range(tensor1.shape[0]):
        img1 = tensor1[i]
        img2 = tensor2[i]

        if img1.shape[0] == 3:
            img1 = np.transpose(img1, (1, 2, 0))
            img2 = np.transpose(img2, (1, 2, 0))

        psnr_val = peak_signal_noise_ratio(img1, img2, data_range=4.42)
        ssim_val = structural_similarity(img1, img2, channel_axis=-1, data_range=4.42)

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

    return np.mean(psnr_list), np.mean(ssim_list)

def train(model,
	  optimizer,
          loss_fn,
          scheduler,
          train_dataloader,
          test_dataloader,
          epochs,
          test_path_list,
          transforms,
          device,
          save_output_path=None):
	
	train_time_start = Timer()
	train_loss_overtime = []
	test_loss_overtime = []
	test_ssim_overtime = []
	test_psnr_overtime = []

	for epoch in tqdm(range(epochs)):
		print(f"===EPOCH: {epoch+1}===\n")

		train_loss, test_loss = 0, 0
		test_ssim, test_psnr = 0, 0

		model.train()

		print("Training...\n")
		for X, y in tqdm(train_dataloader):
			X, y = X.to(device), y.to(device)

			optimizer.zero_grad()

			y_pred = model(X)

			loss = loss_fn(y, y_pred)

			train_loss += loss.item()

			loss.backward()

			optimizer.step()



		train_loss_overtime.append(train_loss/len(train_dataloader))

		model.eval()

		print("Testing...\n")
		with torch.inference_mode():
			for i, (X_test, y_test) in enumerate(tqdm(test_dataloader)):
				X_test, y_test = X_test.to(device), y_test.to(device)

				test_pred = model(X_test)

				test_loss += loss_fn(y_test, test_pred).item()
				ssim, psnr = compute_psnr_ssim(y_test, test_pred)

				test_ssim += ssim
				test_psnr += psnr

				if save_output_path and i==0:
					os.makedirs(save_output_path, exist_ok=True)
					test_image = Image.open(test_path_list[0][1]).convert("RGB")
					test_image = transforms(test_image).unsqueeze(0).to(device)
					saved_image = model(test_image)
					saved_image = saved_image.detach().cpu().clamp(0, 1)
					filename = f"epoch_{epoch:03d}.png"
					full_path = os.path.join(save_output_path, filename)
					save_image(saved_image, full_path)

			test_loss_overtime.append(test_loss/len(test_dataloader))
			test_ssim_overtime.append(test_ssim/len(test_dataloader))
			test_psnr_overtime.append(test_psnr/len(test_dataloader))

		print(f"Training Loss: {train_loss_overtime[-1]:.5f} | Test Loss: {test_loss_overtime[-1]:.5f} | Test SSIM: {test_ssim_overtime[-1]:.5f} | Test PSNR: {test_psnr_overtime[-1]:.5f}\n")

		scheduler.step()

	train_time_end = Timer()

	print_train_time(train_time_start, train_time_end, device)

	return train_loss_overtime, test_loss_overtime, test_ssim_overtime, test_psnr_overtime