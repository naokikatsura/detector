import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pandas as pd
import tensorflow as tf
# CPUコア数
num_threads = os.cpu_count()
print("CPU threads:", num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC

from .DataVisualizer import DataVisualizer

class ModelTrainer():
	"""
	DESCRIPTION: モデル構築クラス
	INPUT:	None
	OUTPUT:	None
	"""
	def __init__(self, config):
		"""
		DESCRIPTION: 初期化関数
			1. コンフィグを読み込み, selfに格納
		INPUT:	- config: コンフィグインスタンス
		OUTPUT:	None
		"""
		# コンフィグ読み込み
		self.config = config
		config.setupModelTrainer()

	def buildModel(self):
		"""
		DESCRIPTION: モデル構築関数
			1. モデル設定
			2. モデルを構築し, selfに格納
			3. モデルサマリの保存
		INPUT:	None
		OUTPUT:	None
		"""

		# ===== 1. モデルの設定 =====
		# リサイズしない場合
		if self.config.RESIZE == False:
			self.model = keras.Sequential([
				# 入力層
				Input(shape=(							
					self.config.IMAGE_WIDTH, 			# 入力画像幅
					self.config.IMAGE_HEIGHT, 			# 入力画像高さ
					self.config.IMAGE_CH,				# 入力画像チャンネル数
				)),

				# Block1
				Conv2D(32,3,padding="same",activation="relu"),
				BatchNormalization(),
				Conv2D(32,3,padding="same",activation="relu"),
				MaxPooling2D(2),

				# Block2
				Conv2D(64,3,padding="same",activation="relu"),
				BatchNormalization(),
				Conv2D(64,3,padding="same",activation="relu"),
				MaxPooling2D(2),

				# Block3
				Conv2D(128,3,padding="same",activation="relu"),
				BatchNormalization(),
				Conv2D(128,3,padding="same",activation="relu"),
				MaxPooling2D(2),

				# Block4
				Conv2D(256,3,padding="same",activation="relu"),
				MaxPooling2D(2),

				GlobalAveragePooling2D(),

				Dense(256, activation="relu"),
				Dropout(0.5),

				Dense(1, activation="sigmoid")
			])
		# リサイズする場合
		elif self.config.RESIZE == True:
			self.model = keras.Sequential([
				# 入力層
				Input(shape=(							
					self.config.RESIZED_IMAGE_WIDTH, 	# 入力画像幅
					self.config.RESIZED_IMAGE_WIDTH,	# 入力画像高さ
					self.config.IMAGE_CH,				# 入力画像チャンネル数
				)),

				# Block1
				Conv2D(32,3,padding="same",activation="relu"),
				BatchNormalization(),
				Conv2D(32,3,padding="same",activation="relu"),
				BatchNormalization(),
				MaxPooling2D(2),

				# Block2
				Conv2D(64,3,padding="same",activation="relu"),
				BatchNormalization(),
				Conv2D(64,3,padding="same",activation="relu"),
				BatchNormalization(),
				MaxPooling2D(2),

				# Block3
				Conv2D(128,3,padding="same",activation="relu"),
				BatchNormalization(),
				Conv2D(128,3,padding="same",activation="relu"),
				BatchNormalization(),
				MaxPooling2D(2),

				# Block4
				Conv2D(256,3,padding="same",activation="relu"),
				BatchNormalization(),
				MaxPooling2D(2),

				GlobalAveragePooling2D(),

				Dense(256, activation="relu"),
				Dropout(0.5),

				Dense(1, activation="sigmoid")
			])

		optimizer = Adam(learning_rate=0.0001)

		# ===== 2. モデルの構築 =====
		self.model.compile(
			optimizer=optimizer,								# 誤差逆伝播
			loss=self.config.LOSS_FUNC,							# 損失関数
			metrics=['accuracy', Precision(), Recall(),	AUC()]	# 訓練時に監視する指標
		)

		# ===== 3. モデルサマリの出力 =====
		model_summary_filepath = os.path.join(
			self.config.LOG_DIR,
			"model_summary_{}.log".format(self.config.START_TIME)
		)
		with open(model_summary_filepath, "w", encoding="utf-8") as fp:
			self.model.summary(print_fn=lambda x: fp.write(x + "\r\n"))

		return self.model

	def setCallbacks(self):
		"""
		DESCRIPTION: 学習時のコールバック郡を定義
			1. ベストモデルを保存する設定をselfに格納
			2. 規定エポック数ごとにモデルを保存する設定をselfに格納
			3. 学習結果が改善しない場合に学習停止する設定をselfに格納
		INPUT:	None
		OUTPUT:	None
		"""
		# ===== 1. ベストモデル（val_loss 改善時）を保存 =====
		model_file_path = os.path.join(
			self.THIS_MODEL_DIR_PATH, 
			self.config.MODEL_BEST_FILENAME
		)
		self.best_model_cb = tf.keras.callbacks.ModelCheckpoint(
			filepath=model_file_path,
			monitor='val_loss',
			save_best_only=True,
			save_weights_only=False,
			verbose=1
		)

		# ===== 2. 10エポック毎のモデルを保存 =====
		def save_every_X_epochs(epoch, logs=None):
			if (epoch + 1) % self.config.TRAIN_PATIENCE == 0:
				model_X_epoch_path = os.path.join(
					self.THIS_MODEL_DIR_PATH, 
					f"model_epoch_{epoch+1:03d}.keras"
				)
				self.model.save(model_X_epoch_path)
				print(f"10エポックごと保存: {model_X_epoch_path}")
		self.save_every_10_cb = tf.keras.callbacks.LambdaCallback(
			on_epoch_end=save_every_X_epochs
		)

		# ===== 3. EarlyStopping（バリデーション損失が改善しない場合に停止）=====
		self.early_stopping = tf.keras.callbacks.EarlyStopping(
			monitor='val_loss', 
			patience=self.config.TRAIN_PATIENCE,
			restore_best_weights=True
		)

	def saveFinalModel(self):
		"""
		DESCRIPTION: 学習終了時のモデルを保存
		INPUT:	None
		OUTPUT:	None
		"""
		# 学習終了後に最終モデル保存
		model_final_path = os.path.join(
			self.THIS_MODEL_DIR_PATH, 
			self.config.MODEL_FINAL_FILENAME
		)
		self.model.save(model_final_path)
		print(f"学習終了後に最終モデル保存: {model_final_path}")

	def saveTrainLog(self, train_history):
		"""
		DESCRIPTION: 学習終了時のモデルを保存
		INPUT:	- train_history: 訓練履歴
		OUTPUT:	None
		"""
		# 訓練ログを保存
		train_log_filepath = os.path.join(
			self.THIS_MODEL_DIR_PATH, 
			self.config.TRAIN_LOG_FILENAME
		)
		pd.DataFrame(train_history.history).to_csv(train_log_filepath, index=False)
		print("学習モデルファイル保存完了")

	def runTraining(self, ds_train, ds_val, class_weight):
		"""
		DESCRIPTION: 学習実行関数
			1. モデル保存ディレクトリ作成
			2. モデル構築
			3. 学習を実行
			4. 学習済みモデル・学習ログを保存

		INPUT:	- ds_train:		訓練用データセット
				- ds_val:		評価用データセット
				- class_weight:	陽性・陰性のクラス重み
		OUTPUT:	None
		"""
		# ===== 1. 保存ディレクトリを作成 =====
		self.THIS_MODEL_DIR_PATH = os.path.join(self.config.MODEL_DIR_PATH, self.config.START_TIME)
		os.makedirs(self.THIS_MODEL_DIR_PATH, exist_ok=True)

		# ===== 2. モデル構築 =====
		self.buildModel()
		# コールバック郡を定義
		self.setCallbacks()

		# ===== 3. 学習を実行 =====
		train_history = self.model.fit(
			ds_train,
			validation_data=ds_val,
			epochs=self.config.NUM_EPOCH,
			class_weight=class_weight,
			callbacks=[self.early_stopping, self.best_model_cb, self.save_every_10_cb],
			verbose=1
		)

		# ===== 4. 学習済みモデル・ログを保存 =====
		# 学習終了時のモデルを保存
		self.saveFinalModel()
		# 学習履歴を保存
		self.saveTrainLog(train_history)
		# 学習履歴をグラフにして保存
		data_visualizer = DataVisualizer()
		train_graph_flename = "training_metrics_{}.png".format(self.config.START_TIME)
		data_visualizer.plotTrainingHistory(
			train_history=train_history,
			output_dir_path=self.THIS_MODEL_DIR_PATH, 
			output_filename=train_graph_flename
		)