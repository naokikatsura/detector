import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import confusion_matrix

from .DataVisualizer import DataVisualizer

class ObjectDetector:
	"""
	DESCRIPTION: 物体検知クラス
	INPUT:	None
	OUTPUT:	None
	"""
	def __init__(self, config):
		"""
		DESCRIPTION: コンフィグを読み込み, selfに格納
		INPUT:	- config: コンフィグインスタンス
		OUTPUT:	None
		"""
		# コンフィグ読み込み
		self.config = config
		config.setupObjectDetector()

	def loadTrainedModel(self):
		"""
		DESCRIPTION: 訓練済みモデルを読み込み, selfに格納
		INPUT:	None
		OUTPUT:	None
		"""
		# 訓練済みモデル読み込み
		trained_model_filepath = os.path.join(
			self.config.TRAINED_MODEL_DIR_PATH,
			self.config.TRAINED_MODEL_FILENAME
		)
		self.model = tf.keras.models.load_model(trained_model_filepath)
		print("モデル読み込み完了")

	def loadInputImages(self, input_dir_path, num_images, true_label):
		"""
		DESCRIPTION: 入力画像ロード関数
			1. 入力画像ファイルパスを取得
			2. 画像を読み込んで配列に格納
		INPUT:	- input_dir_path: 入力画像フォルダ
		OUTPUT:	- input_images:	入力画像配列
				- valid_paths:	入力画像のパス配列
		"""

		# ===== 1. 画像ファイルパスを取得 =====
		img_paths = [
			os.path.join(input_dir_path, f)
			for f in os.listdir(input_dir_path)
			if f.lower().endswith((".png", ".jpg", ".jpeg"))
		]

		# ===== 2. 画像を読み込んで配列に格納 =====
		input_images = []
		valid_paths = []
		true_labels = []

		def _loadImage(img_path):
			"""
			DESCRIPTION: 画像読みこみ関数
				1. 画像を読み込む
				2. 正規化して画素値を0-1に変換
			INPUT:	-path:	画像ファイルのパス
			OUTPUT: -image:	正規化済み画像配列
			"""
			# 画像読み込み
			img = tf.io.read_file(img_path)
			img = tf.image.decode_png(img, channels=self.config.IMAGE_CH)

			# リサイズ
			if self.config.RESIZE == False:
				img = tf.image.resize(img,
					(self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH)
				)
			
			else:
				img = tf.image.resize(img,
					(self.config.RESIZED_IMAGE_HEIGHT, self.config.RESIZED_IMAGE_WIDTH),
					method=self.config.RESIZE_METHOD
				)

			# 正規化
			img = tf.cast(img, tf.float32) / 255.0

			return img
		
		for img_path in img_paths[:num_images]:

			# 画像を読み込み			
			img = _loadImage(img_path=img_path)

			# 画像配列を配列に追加
			input_images.append(img.numpy())
			valid_paths.append(img_path)
			true_labels.append(true_label)

		# numpy配列に変換
		input_images = np.array(input_images)
		valid_paths  = np.array(valid_paths)
		true_labels  = np.array(true_labels)

		print("ラベル{}データ{}件読み込み完了".format(true_label, num_images))

		return input_images, valid_paths, true_labels

	def detectObject(self, input_images, image_paths, true_labels):
		"""
		DESCRIPTION: 物体検知実行関数
			1. 推論を実行
			2. 推論結果をまとめる
			3. 推論結果をデータフレームに変換
		INPUT:	- input_images: 入力画像配列
		OUTPUT:	- df_results:	推論結果データフレーム(ファイルパス，ラベル，推論結果)
		"""
		# ===== 1. 推論を実行 =====
		probs = self.model.predict(
			input_images,
			batch_size=self.config.BATCH_SIZE, 
			verbose=0
		)

		# ===== 2. 推論結果を集約 =====
		results = []

		for img_path, true_label, prob in zip(image_paths, true_labels, probs):
			pred_label = 1 if prob[0] > 0.5 else 0

			results.append((img_path, true_label, pred_label, prob[0]))

		# ===== 3. 推論結果をデータフレームに変換 =====
		df_results = pd.DataFrame(results, columns=["filepath", "true_label", "pred_label", "probability"])

		print("ラベル{}データ検知完了".format(true_labels[0]))
		return df_results
	
	def saveResultsCSV(self, df_results):
		"""
		DESCRIPTION: 検知結果保存関数
		INPUT:	- results:	推論結果(ファイルパス，ラベル，推論結果)
		OUTPUT:	None 
		"""
		# ===== 1. 結果CSVを保存 =====
		# 結果CSVのファイルパスを設定
		results_csv_filename = "detection_results_{}.csv".format(self.config.START_TIME)
		results_csv_path = os.path.join(self.THIS_RESULTS_DIR_PATH, results_csv_filename)
		# 結果CSVを保存
		df_results.to_csv(results_csv_path, index=False)
		print("CSV保存:", results_csv_path)
		
	def saveEvaluationParameters(self, df_results, visualizer):
		"""
		DESCRIPTION: 検知結果保存関数
			1. 正解ラベルと推論ラベルを配列に格納
			2. 混合行列を計算
			3. 性能評価パラメータを計算
			4. 性能評価パラメータを保存
			5. 混合行列を保存
		INPUT:	- results:	推論結果(ファイルパス，ラベル，推論結果)
				- visualizer: 結果可視化クラス
		OUTPUT:	None
		"""
		# ===== 1. 正解ラベルと推論ラベルを配列に格納 =====
		y_true = df_results["true_label"].to_numpy()	# 正解ラベル
		y_pred = df_results["pred_label"].to_numpy()	# 推論ラベル

		# ===== 2. 混同行列を計算 ===== 
		cm = confusion_matrix(y_true, y_pred, labels=[0,1])

		# ===== 3. 性能評価パラメータを計算 =====
		TN, FP, FN, TP = cm.ravel()
		accuracy = (TP + TN) / (TP + TN + FP + FN)
		precision = TP / (TP + FP) if (TP + FP) > 0 else 0
		recall = TP / (TP + FN) if (TP + FN) > 0 else 0
		f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

		# ===== 4. 性能評価パラメータを保存 =====
		txt_filename = "evaluation_{}.txt".format(self.config.START_TIME)
		txt_path = os.path.join(self.THIS_RESULTS_DIR_PATH, txt_filename)

		with open(txt_path, "w") as f:

			f.write(f"TP: {TP}\n")
			f.write(f"TN: {TN}\n")
			f.write(f"FP: {FP}\n")
			f.write(f"FN: {FN}\n\n")

			f.write(f"Accuracy: {accuracy:.4f}\n")
			f.write(f"Precision: {precision:.4f}\n")
			f.write(f"Recall: {recall:.4f}\n")
			f.write(f"F1 Score: {f1:.4f}\n")

		print("評価保存:", txt_path)

		# ===== 5. 混合行列を保存 =====
		cm_filename = "confusion_matirix_{}".format(self.config.START_TIME)
		visualizer.plotConfusionMatrix(
			confusion_matrix=cm, 
			output_dir_path=self.THIS_RESULTS_DIR_PATH, 
			output_filename=cm_filename)

	def runDetection(self):
		"""
		DESCRIPTION: 物体検知実行関数
			1. 訓練モデル読み込み
			2. 入力画像を読み込み
			3. 検知実行
			4. 結果を保存
		INPUT:	None
		OUTPUT:	None
		"""
		# ===== 1. 訓練モデル読み込み ===== 
		self.loadTrainedModel()

		# ===== 2. 入力画像を読み込み =====
		# 陽性データの読み込み
		input_images_pos, image_paths_pos, true_labels_pos = self.loadInputImages(
			input_dir_path=self.config.INPUT_POS_DIR_PATH,
			num_images=self.config.NUM_INPUT_POS,
			true_label=1
		)
		# 陰性データの読み込み
		input_images_neg, image_paths_neg, true_labels_neg = self.loadInputImages(
			input_dir_path=self.config.INPUT_NEG_DIR_PATH,
			num_images=self.config.NUM_INPUT_NEG,
			true_label=0
		)

		# ===== 3. 検知実行 =====
		# 陽性データの検知
		df_results_pos = self.detectObject(
			input_images=input_images_pos, 
			image_paths=image_paths_pos,
			true_labels=true_labels_pos
		)
		# 陰性データの検知
		df_results_neg = self.detectObject(
			input_images=input_images_neg, 
			image_paths=image_paths_neg,
			true_labels=true_labels_neg
		)
		# 陽性・陰性の検知結果を1つのデータフレームに統合
		df_results_all = pd.concat([df_results_pos, df_results_neg], ignore_index=True)

		# ===== 4. 結果を保存 ===== 
		self.THIS_RESULTS_DIR_PATH = os.path.join(
			self.config.OUTPUT_DIR_PATH,
			self.config.START_TIME
		)
		os.makedirs(self.THIS_RESULTS_DIR_PATH, exist_ok=True)
		data_visualizer = DataVisualizer()
		
		# 推論結果をCSVに保存
		self.saveResultsCSV(df_results=df_results_all)
		
		# 混合行列と性能評価パラメータを保存
		self.saveEvaluationParameters(
			df_results=df_results_all,
			visualizer=data_visualizer
		)