import mxnet as mx
from mxnet.gluon.model_zoo import vision
import time
import os

batch_shape = (1, 3, 224, 224)
resnet18 = vision.resnet18_v2(pretrained=True)
resnet18.hybridize()
resnet18.forward(mx.nd.zeros(batch_shape))
resnet18.export('resnet18_v2')
sym, arg_params, aux_params = mx.model.load_checkpoint('resnet18_v2', 0)

# Create sample input
input = mx.nd.zeros(batch_shape)

# Execute with MXNet
os.environ['MXNET_USE_TENSORRT'] = '0'
# -------------------------------------------------------------------------
executor = sym.simple_bind(
    ctx=mx.gpu(0),
    data=batch_shape,
    grad_req='null',
    force_rebind=True)
executor.copy_params_from(arg_params, aux_params)
# -------------------------------------------------------------------------

# Warmup
print('Warming up MXNet')
for i in range(0, 10):
    y_gen = executor.forward(is_train=False, data=input)
    y_gen[0].wait_to_read()

# Timing
print('Starting MXNet timed run')
start = time.time()
for i in range(0, 10000):
    y_gen = executor.forward(is_train=False, data=input)
    y_gen[0].wait_to_read()
print(time.time() - start)

# Execute with TensorRT
print('Building TensorRT engine')
os.environ['MXNET_USE_TENSORRT'] = '1'
# -------------------------------------------------------------------------
arg_params.update(aux_params)
all_params = dict([(k, v.as_in_context(mx.gpu(0))) for k, v in arg_params.items()])
executor = mx.contrib.tensorrt.tensorrt_bind(
    sym,
    all_params=all_params,
    ctx=mx.gpu(0),
    data=batch_shape,
    grad_req='null',
    force_rebind=True)
# -------------------------------------------------------------------------

#Warmup
print('Warming up TensorRT')
for i in range(0, 10):
    y_gen = executor.forward(is_train=False, data=input)
    y_gen[0].wait_to_read()

# Timing
print('Starting TensorRT timed run')
start = time.time()
for i in range(0, 10000):
    y_gen = executor.forward(is_train=False, data=input)
    y_gen[0].wait_to_read()
print(time.time() - start)
                                  
