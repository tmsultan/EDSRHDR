from importlib import import_module
#from dataloader import MSDataLoader

from IPython.core import debugger 
breakpoint = debugger.set_trace

# Import pytorch data structures
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset

# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

class Data:
    def __init__(self, args):
        self.loader_train = None

        # Go here if want to train
        if not args.test_only:
            datasets = []
            for d in args.data_train:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                datasets.append(getattr(m, module_name)(args, name=d))

            self.loader_train = dataloader.DataLoader(
                MyConcatDataset(datasets),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )

        self.loader_test = []
        for d in args.data_test:
            if d in ['Set5', 'Set14', 'B100', 'Urban100']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            else:

                # GO here since data_test == 'Demo
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'

                # It imports all the modules from the file in the directory corresponding to d
                # here - d = demo.py --> imports files from src/data.py 
                # Can also import from other files in directory e.g. common.py or div2k.py!
                m = import_module('data.' + module_name.lower())
                

                # Assign to testset all attributes in demo.Demo class 
                # testset type is data.demo.Demo
                # passed into it we have the idx_scale and other args
                testset = getattr(m, module_name)(args, train=False, name=d)

            self.loader_test.append(
                # Call Data loader - Load demo data into data loader
                # Built in pytorch function
                dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                )
            )
