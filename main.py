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

# Load checkpoints here
# Checkpoints is a class 
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
            
            # Defining a lot of attributes of loader class, including where to load test images
            # But we have not actually loaded data --> only in t.test()
            loader = data.Data(args)
        
            # model defined in src/model/__init__.py
            # important uses from args: precision, scale, cpu, m_GPUs, 
                # define method that creates forward model 
                # Don't need detail for now
            _model = model.Model(args, checkpoint)
            
        
            # model defined in src/loss/__init__.py
            # define loss function - default set to L1 at args (option.py)
                # 

            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            
            # import class directly from trainer.py defined in src (same directory)
            # checkpoint - defined as cpk inside
            # Creating test, and train class on t -- but have not called the 
            t = Trainer(args, loader, _model, _loss, checkpoint)
            
            
            while not t.terminate():
                t.train()
                t.test()
                
            
            checkpoint.done()

if __name__ == '__main__':
    main()
