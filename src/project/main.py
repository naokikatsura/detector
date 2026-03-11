from .module.Config import Config
from .module.TimeLogger import TimeLogger
from .module.DatasetGenerator import DatasetGenerator
from .module.ModelTrainer import ModelTrainer
from .module.ObjectDetector import ObjectDetector

def main():
	"""
	DESCRIPTION: メイン関数
	INPUT:	None
	OUTPUT:	None
	"""
	# コンフィグインスタンス作成
	config = Config()
	config.runConfig()

	time_logger = TimeLogger(config=config)

	if config.RUN_TRAIN == True:

		# ===== 学習データセットを作成 =====		
		time_logger.getStartTime("Dataset Generation")
		# 学習データセット作成インスタンスを作成
		data_loader = DatasetGenerator(config=config)
		# 学習データセットを作成
		ds_train, ds_val, class_weight = data_loader.runDatasetgeneration()
		time_logger.getEndTime("Dataset Generation")

		# ===== モデルを訓練 =====
		time_logger.getStartTime("Model Training")
		# 学習実行インスタンスを作成
		trainer = ModelTrainer(config=config)
		# 学習を実行
		trainer.runTraining(
			ds_train=ds_train, 
			ds_val=ds_val, 
			class_weight=class_weight
		)
		time_logger.getEndTime("Model Training")
	
	if config.RUN_DETECTION == True:

		# ===== 物体検知 =====
		time_logger.getStartTime("Object Detection")
		# 物体検知インスタンスを作成
		object_detactor = ObjectDetector(config=config)
		object_detactor.runDetection()
		time_logger.getEndTime("Object Detection")

	# ===== 実行時間ログを保存 =====
	time_logger.printLogSummary()
	time_logger.saveLogSummary()