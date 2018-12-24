from utils import *
from YOLO import *

args = yolo_Parser()
yolo = YOLO(args)

sym, arg_params, aux_params = mx.model.load_checkpoint('export/YOLO_export', 0)
batch_shape = (1, 3, 320, 512)
input = nd.zeros(batch_shape).as_in_context(yolo.ctx[0])
# -------------------------------------------------------------------------
'''
print('Warming up YOLO.net')
for i in range(0, 10):
    y_gen = yolo.net(input)
    y_gen[0][0].wait_to_read()

print('Starting YOLO.net timed run')
start = time.time()
for i in range(0, 1000):
    y_gen = yolo.net(input)
    y_gen[0][0].wait_to_read()
print(time.time() - start)
'''
# Execute with MXNet
os.environ['MXNET_USE_TENSORRT'] = '0'
# -------------------------------------------------------------------------

executor = sym.simple_bind(
    ctx=mx.gpu(1),
    data=batch_shape,
    grad_req='null',
    force_rebind=True)
executor.copy_params_from(arg_params, aux_params)

out = executor.forward(is_train=False, data=input)
batch_out = out[:5]
print(batch_out)
'''
# -------------------------------------------------------------------------

# Warmup
print('Warming up MXNet')
for i in range(0, 10):
    y_gen = executor.forward(is_train=False, data=input)
    y_gen[0].wait_to_read()


# Timing
print('Starting MXNet timed run')
start = time.time()
for i in range(0, 1000):
    y_gen = executor.forward(is_train=True, data=input)
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
    ctx=mx.gpu(1),
    data=batch_shape,
    grad_req='null',
    force_rebind=True)



# -------------------------------------------------------------------------
#Warmup
print('Warming up TensorRT')
for i in range(0, 10):
    y_gen = executor.forward(is_train=False, data=input)
    y_gen[0].wait_to_read()
    a = y_gen[0].asnumpy()
    print(a)

# Timing
print('Starting TensorRT timed run')
start = time.time()
for i in range(0, 1000):
    y_gen = executor.forward(is_train=False, data=input)
    y_gen[0].wait_to_read()
print(time.time() - start)
'''
