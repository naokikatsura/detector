import os
import matplotlib.pyplot as plt

class DataVisualizer:
	def __init__(self):
		pass

	def plotTrainingHistory(self, train_history, output_dir_path, output_filename):
		"""
		DESCRIPTION: 訓練履歴描画関数
		INPUT:	- train_history:	訓練履歴
				- output_dir_path:	訓練履歴保存先
				- output_filenam:	訓練履歴ファイル名
		OUTPUT:	None
		"""
		history = train_history.history
		epochs = range(1, len(history["loss"]) + 1)

		plt.figure(figsize=(12,8))

		# Loss
		plt.subplot(2,2,1)
		plt.plot(epochs, history["loss"], label="train")
		plt.plot(epochs, history["val_loss"], label="val")
		plt.ylim(0, 0.5)
		plt.title("Loss")
		plt.legend()

		# Accuracy
		plt.subplot(2,2,2)
		plt.plot(epochs, history["accuracy"], label="train")
		plt.plot(epochs, history["val_accuracy"], label="val")
		plt.title("Accuracy")
		plt.legend()

		# Precision
		plt.subplot(2,2,3)
		plt.plot(epochs, history["precision"], label="train")
		plt.plot(epochs, history["val_precision"], label="val")
		plt.title("Precision")
		plt.legend()

		# Recall
		plt.subplot(2,2,4)
		plt.plot(epochs, history["recall"], label="train")
		plt.plot(epochs, history["val_recall"], label="val")
		plt.title("Recall")
		plt.legend()

		plt.tight_layout()

		save_path = os.path.join(output_dir_path, output_filename)
		plt.savefig(save_path)
		plt.close()

		print("グラフ保存:", save_path)

	def plotConfusionMatrix(self, confusion_matrix, output_dir_path, output_filename):
		"""
		DESCRIPTION: 物体検知混合行列描画関数
		INPUT:	- confusion_matrix:	混合行列
				- output_dir_path:	混合行列保存先
				- output_filenam:	混合行列ファイル名
		OUTPUT:	None
		"""		
		# 混同行列グラフ
		plt.figure(figsize=(5,5))
		plt.imshow(confusion_matrix, cmap="Blues")
		plt.colorbar()

		labels = ["Negative", "Positive"]

		plt.xticks([0,1], labels)
		plt.yticks([0,1], labels)

		plt.xlabel("Predicted")
		plt.ylabel("True")
		plt.title("Confusion Matrix")

		for i in range(2):
			for j in range(2):
				plt.text(j, i, confusion_matrix[i,j],
						ha="center",
						va="center",
						color="black",
						fontsize=14)

		img_path = os.path.join(output_dir_path, output_filename)
		plt.tight_layout()
		plt.savefig(img_path)
		plt.close()
	