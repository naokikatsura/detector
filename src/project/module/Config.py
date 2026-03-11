import os
from datetime import datetime
from dataclasses import dataclass

class Config:
	"""
	DESCRIPTION: コンフィグクラス
	INPUT:	None
	OUTPUT:	None
	"""
	def __init__(self):
		# ===== コード実行パラメータ =====
		# モデル訓練を行うかどうか
		self.RUN_TRAIN = True
		# 検知を行うかどうか
		self.RUN_DETECTION = False
		# 画像のリサイズを行うかどうか
		self.RESIZE = True

		# ===== 画像パラメータ =====
		# 入力画像パラメータ
		self.IMAGE_WIDTH  = 320
		self.IMAGE_HEIGHT = 320
		self.IMAGE_CH     = 3
		# リサイズ後の入力画像パラメータ
		if self.RESIZE:
			self.RESIZED_IMAGE_WIDTH  = 128
			self.RESIZED_IMAGE_HEIGHT = 128

		# バッチサイズ
		self.BATCH_SIZE = 64


	def setupModelTrainer(self):
		"""
		DESCRIPTION: ModelBuiderクラスで使用するパラメータを設定
		INPUT:	None
		OUTPUT:	None
		"""

		# 1. 畳み込み層のモデルパラメータ
		self.NUM_FILTER_1 	= 32
		self.KERNEL_SIZE_1 	= 3
		self.PADDING_1		= "same"
		self.ACTIVATION_1 	= "relu"

		# 2. 畳み込み層のモデルパラメータ
		self.NUM_FILTER_2 	= 64
		self.KERNEL_SIZE_2 	= 3
		self.PADDING_2		= "same"
		self.ACTIVATION_2 	= "relu"

		# 3. 畳み込み層のモデルパラメータ
		self.NUM_FILTER_3 	= 128
		self.KERNEL_SIZE_3 	= 3
		self.PADDING_3		= "same"
		self.ACTIVATION_3 	= "relu"

		# 全結合層のモデルパラメータ
		self.DENSE_SIZE_2 = 1024
		self.ACTIVATION_2 = "relu"

		# 全結合層のモデルパラメータ
		self.DENSE_SIZE_3 = 128
		self.ACTIVATION_3 = "relu"

		# 出力層
		self.DENSE_SIZE_4 = 1
		self.ACTIVATION_4 = "sigmoid"

		# モデルコンパイルのパラメータ
		self.LOSS_FUNC = "binary_crossentropy"

		# モデル保存先
		self.MODEL_DIR_PATH = "PATH/TO/detector/model/"

		# モデルファイル名
		self.MODEL_BEST_FILENAME = "model_best.keras"
		self.MODEL_FINAL_FILENAME = "model_final.keras"
		
		# 訓練ログファイル名
		self.TRAIN_LOG_FILENAME = "train_log.csv"

		# 学習エポック数
		self.NUM_EPOCH = 10

		# アーリーストッピング
		self.TRAIN_PATIENCE = 5

	def setupDatasetGenerator(self):
		"""
		DESCRIPTION: DatasetGeneratorクラスで使用するパラメータを設定
		INPUT:	None
		OUTPUT:	None
		"""
		# ファイル選択・シャッフルに使用するランダムシード
		self.RANDOM_SEED = 100
		# 陽性・陰性データファイル格納先
		self.POS_DIR_PATH = "PATH/TO/detector/data/train/positive_formatted"
		self.NEG_DIR_PATH = "PATH/TO/detector/data/train/negative_formatted"
		# 陽性・陰性データ数
		self.NUM_POS = 8192*4
		self.NUM_NEG = 8192*4
		# 訓練データの割合
		self.RATIO_TRAIN = 0.8
		
		# リサイズ方法
		if self.RESIZE:
			self.RESIZE_METHOD = "area"

	def setupObjectDetector(self):
		"""
		DESCRIPTION: ObjectDetectorクラスで使用するパラメータを設定
		INPUT:	None
		OUTPUT:	None
		"""
		# 訓練済みモデル保存先
		self.TRAINED_MODEL_DIR_PATH = "PATH/TO/detector/model/YYYYMMDD_HHMMSS"
		# 訓練済みモデルファイル名
		self.TRAINED_MODEL_FILENAME = "model_best.keras"
		# 入力ファイル格納先
		self.INPUT_POS_DIR_PATH = "PATH/TO/detector/data/test/positive_formatted"
		self.INPUT_NEG_DIR_PATH = "PATH/TO/detector/data/test/negative_formatted"
		# 入力ファイル数
		self.NUM_INPUT_POS = 1000
		self.NUM_INPUT_NEG = 1000
		# 結果出力先
		self.OUTPUT_DIR_PATH = "PATH/TO/detector/output"


	def saveConfig(self):
		"""
		DESCRIPTION: パラメータをtxtで保存
		INPUT:	None
		OUTPUT:	None
		"""
		# 時刻取得
		self.START_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
		# ログファイルに説明を追加
		description = "" #input("Description: ")
		# ログファイル名を設定
		output_filename = "./config_{}_{}.log".format(self.START_TIME, description)
		# ログファイル出力先
		self.LOG_DIR = "PATH/TO/detector/DETECTOR/log/{}".format(self.START_TIME)
		os.makedirs(self.LOG_DIR, exist_ok=True)
		# ログファイルパスを設定
		output_file_path = self.LOG_DIR + output_filename

		# ログファイルを出力
		with open(output_file_path, "w", encoding="utf-8") as f:
			for key, value in self.__dict__.items():
				f.write(f"{key} = {value}\n")

	def runConfig(self):
		"""
		DESCRIPTION: コンフィグを設定してtxtで保存
		INPUT:	None
		OUTPUT:	None
		"""
		# 各クラスのセットアップ
		self.setupDatasetGenerator()
		self.setupModelTrainer()
		self.setupObjectDetector()

		# コンフィグを保存
		self.saveConfig()

