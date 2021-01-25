import json
import os


class PyJSON(object):
	def __init__(self, data):
		if type(data) is str:
				data = json.loads(data)
		self.convert_json(data)

	def convert_json(self, data):
		self.__dict__ = {}
		for key, value in data.items():
			if type(value) is dict:
					value = PyJSON(value)
			self.__dict__[key] = value

	def __setitem__(self, key, value):
		self.__dict__[key] = value

	def __getitem__(self, key):
		return self.__dict__[key]


def parse_config_from_json(config_file):
	"""Read model config params from json config file

	Args:
		json_file: a path to the json config file

	Returns:
		config: a PyJSON object containing configuration params
	"""

	with open(config_file) as file:
		file_data = file.read().replace('\n', '')
		config = PyJSON(data = file_data)

		return config


