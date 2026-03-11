import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
from PIL import Image
import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split


class DatasetGenerator():
	"""
	DESCRIPTION: データセット作成クラス
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
		config.setupDatasetGenerator()

	def generateDataFrame(self, rng, pos_dir_path, neg_dir_path, 
			num_pos, num_neg, ratio_train
		):
		"""
		DESCRIPTION: データフレーム作成関数
			1. 陽性・陰性ファイルパスを読み込み
			2. ファイルパスからランダムに一部選択
			3. 選択したファイルパスをデータフレームに格納して出力
		INPUT:	- rng: 乱数生成器
				- pos_dir_path:	陽性データ格納ディレクトリ
				- neg_dir_path:	陰性データ格納ディレクトリ
				- num_pos: 陽性データ数
				- num_neg: 陰性データ数
		OUTPUT:	- df_selected: データ
		"""
		# ファイルパスを読み込み
		pos_path_all = [
			f for f in os.listdir(pos_dir_path)
			if f.lower().endswith((".png", ".jpg", ".jpeg"))
		]
		neg_path_all = [
			f for f in os.listdir(neg_dir_path)
			if f.lower().endswith((".png", ".jpg", ".jpeg"))
		]
		
		# 最大件数を制限
		self.num_pos = min(len(pos_path_all), num_pos)
		self.num_neg = min(len(neg_path_all), num_neg)
		print(f"使用ファイル数: 正例={self.num_pos}, 負例={self.num_neg}, 合計={self.num_pos+self.num_neg}")

		# ランダムにサンプリングして結合
		data_path_selected = (
			[(os.path.join(pos_dir_path, f), 1) for f in rng.choice(pos_path_all, self.num_pos, replace=False)] +
			[(os.path.join(neg_dir_path, f), 0) for f in rng.choice(neg_path_all, self.num_neg, replace=False)]
		)

		# DataFrameに変換
		df_selected = pd.DataFrame(data_path_selected, columns=["filepath", "label"])
		df_selected["filepath"] = df_selected["filepath"].str.replace("\\", "/", regex=False)

		df_train, df_val = train_test_split(
			df_selected,
			train_size=ratio_train,
			stratify=df_selected["label"],
			random_state=rng.integers(1e9)
		)

		return df_train.reset_index(drop=True), df_val.reset_index(drop=True)

	def calcClassWeight(self):
		"""
		DESCRIPTION: クラス重み計算関数
		INPUT:	None
		OUTPUT: None
		"""
		num_total = self.num_pos + self.num_neg
		class_weight = {0: num_total/(2*self.num_neg), 1: num_total/(2*self.num_pos)}
		print(f"クラス重み: {class_weight}")
		return class_weight

	def generateDataset(self, df):
		"""
		DESCRIPTION: データセット作成関数
		INPUT:	-df:	データフレーム[filename], [label]
		OUTPUT: -ds:	データセット
		"""
		image_paths = df["filepath"].values
		labels = df["label"].values

		def _loadImage(path, label):
			"""
			DESCRIPTION: 画像読みこみ関数
				1. 画像を読み込む
				2. 正規化して画素値を0-1に変換
			INPUT:	-path:	画像ファイルのパス
					-label:	画像ファイルのラベル（陽性/陰性）
			OUTPUT: -image:	正規化済み画像配列
					-label:	画像ファイルのラベル（陽性/陰性）
			"""
			# 画像読み込み
			img = tf.io.read_file(path)
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

			return img, label

		ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

		ds = ds.shuffle(
			buffer_size=len(image_paths),
			seed=self.config.RANDOM_SEED,
			reshuffle_each_iteration=True
		)

		ds = ds.map(
			lambda x, y: _loadImage(x, y),
			num_parallel_calls=tf.data.AUTOTUNE
		)

		ds = ds.batch(self.config.BATCH_SIZE)
		ds = ds.prefetch(tf.data.AUTOTUNE)

		return ds

	def runDatasetgeneration(self):
		"""
		DESCRIPTION: データセット作成関数
		INPUT:	None
		OUTPUT:	- ds_train:		訓練用データセット
				- ds_val:		評価用データセット
				- class_weight:	クラス重み
		"""
		# ランダムシードを固定して乱数生成器を作成
		rng = np.random.default_rng(seed=self.config.RANDOM_SEED)
		
		# 陽性・陰性データセットを作成
		df_train, df_val = self.generateDataFrame(
			rng=rng,
			pos_dir_path=self.config.POS_DIR_PATH,
			neg_dir_path=self.config.NEG_DIR_PATH,
			num_pos=self.config.NUM_POS,
			num_neg=self.config.NUM_NEG,
			ratio_train=self.config.RATIO_TRAIN
		)

		# データ数に応じてクラス重みを計算
		class_weight = self.calcClassWeight()

		# 学習用データセットを作成
		ds_train = self.generateDataset(df_train)
		ds_val   = self.generateDataset(df_val)
		
		return ds_train, ds_val, class_weight