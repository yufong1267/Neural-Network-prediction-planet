import os

class GetFileList():

	def FileList(self ,path):
		result_dir = next(os.walk(path))[1]  #獲得資料夾
		result_file = next(os.walk(path))[2]  #獲得檔案

		return result_dir , result_file