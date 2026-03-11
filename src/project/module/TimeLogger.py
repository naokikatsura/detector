import os
import time
from datetime import datetime

class TimeLogger:
	"""
	DESCRIPTION: コンフィグクラス
	INPUT:	None
	OUTPUT:	None
	"""
	def __init__(self, config):
		"""
		DESCRIPTION: コンフィグをselfに格納
		INPUT:	None
		OUTPUT:	None
		"""
		self.config = config
		self.logs = {}

	def getStartTime(self, process_name):
		"""
		DESCRIPTION: 処理開始時間を取得する関数
		INPUT:	- process_name: 実行処理の名前
		OUTPUT:	None
		"""
		# 開始時間を記録
		self.logs[process_name] = {
			"start": time.time(),
			"end": None,
			"elapsed": None
		}

	def getEndTime(self, process_name):
		"""
		DESCRIPTION: 処理終了時間を取得する関数
		INPUT:	- process_name: 実行処理の名前
		OUTPUT:	None
		"""
		# エラー（記録なし）ハンドリング
		if process_name not in self.logs:
			raise ValueError(f"{process_name} has not been started")
		# 終了時間を記録
		self.logs[process_name]["end"] = time.time()
		# 処理時間を記録
		self.logs[process_name]["elapsed"] = (
			self.logs[process_name]["end"] - self.logs[process_name]["start"]
		)

	def printLogSummary(self):
		"""
		DESCRIPTION: 処理時間をまとめて標準出力
		INPUT:	None
		OUTPUT:	None
		"""
		print("\n===== Process Time Summary =====")
		# 記録した処理だけ標準出力する
		for process_name, log in self.logs.items():
			elapsed = log["elapsed"]
			if elapsed is None:
				elapsed_str = "Running"
			else:
				elapsed_str = f"{elapsed:.3f} sec"

			print(f"{process_name:20s} : {elapsed_str}")

	def saveLogSummary(self):
		"""
		DESCRIPTION: 処理時間をまとめて.logに保存
		INPUT:	None
		OUTPUT:	None
		"""
		# ログファイル名を設定
		log_filename = "process_time_{}.log".format(self.config.START_TIME)
		log_filepath = os.path.join(
			self.config.LOG_DIR,
			log_filename
		)
		# ログファイルに書き込み
		with open(log_filepath, "w") as f:
			f.write("Process,Start,End,Elapsed(sec)\n")
			# 記録した処理の数だけ書き込む
			for process_name, log in self.logs.items():
				start = datetime.fromtimestamp(log["start"])
				end = log["end"]
				elapsed = log["elapsed"]

				if end:
					end = datetime.fromtimestamp(end)

				f.write(f"{process_name},{start},{end},{elapsed}\n")