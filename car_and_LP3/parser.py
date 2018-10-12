import argparse
def Parser():
	parser = argparse.ArgumentParser(prog="python YOLO.py")
	parser.add_argument("version", help="v1")
	parser.add_argument("mode", help="train or valid or video")
	parser.add_argument("-t", "--topic", help="ros topic to subscribe", dest="topic", default="")
	parser.add_argument("--radar", help="show radar plot", dest="radar", default=False, type=bool)
	parser.add_argument("--show", help="show processed image", dest="show", default=True, type=bool)
	parser.add_argument("--gpu", help="gpu index", dest="gpu", default="0")
	return parser.parse_args()

