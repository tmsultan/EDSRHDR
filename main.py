import torch
from IPython.core import debugger 
breakpoint = debugger.set_trace

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    
    global model

    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            
            # Load Data --> Go to data.Data
            # Class Data defined in /src/loader --> how do we know it is calling that?
                # It calls using import data --> data is a package now due to __init__.py 
                # Any .py file can be imported 
            # Why is Data class defined in .py file?
            loader = data.Data(args)
            breakpoint()
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()
            
            checkpoint.done()

if __name__ == '__main__':
    main()
