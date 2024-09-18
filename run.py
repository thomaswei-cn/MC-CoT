from utils.register import registry
import language_engine
import method
import visual_engine
import utils
from utils.options import get_parser
from utils.dataset_loader import DatasetLoader


if __name__ == '__main__':
    args = get_parser()
    dataset = DatasetLoader(args)
    method = registry.get_class(args.method)(dataset, args)
    method.run()
