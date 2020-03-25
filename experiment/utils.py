import pickle


class Utils(object):
	def __init__(self, args):
		super(Utils, self).__init__()
		self.args = args


	@classmethod
	def save_obj(cls, obj, based_dir, name):
		with open(based_dir + name + '.pkl', 'wb') as f:
			pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


	@classmethod
	def load_obj(cls, based_dir, name):
		with open(based_dir + name + '.pkl', 'rb') as f:
			return pickle.load(f)