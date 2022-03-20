import hls4ml
from qkeras import *
from keras import *
import pickle as pkl

CNV_OUT_CH_POOL = [(64, False), (64, True), (128, False), (128, True), (256, False), (256, False)]
INTERMEDIATE_FC_FEATURES = [(256, 512), (512, 512)]
LAST_FC_IN_FEATURES = 512
LAST_FC_PER_OUT_CH_SCALING = False
POOL_SIZE = 2
KERNEL_SIZE = (3, 3)
WEIGHT_BIT_WIDTH = 2
ACT_BIT_WIDTH = 2
IN_BIT_WIDTH = 8
IN_CHANNELS = 3
NUM_CLASSES = 10

model = keras.models.Sequential()

model.add(keras.layers.Input((32,32,3)))
for out_ch, is_pool_enabled in CNV_OUT_CH_POOL:
    model.add((QConv2D(out_ch, KERNEL_SIZE, use_bias=False, kernel_quantizer=ternary())))
    model.add(keras.layers.BatchNormalization())
    if is_pool_enabled:
        model.add(keras.layers.MaxPooling2D())

for in_features, out_features in INTERMEDIATE_FC_FEATURES:
    model.add(QDense(out_features, use_bias=False, kernel_quantizer=ternary()))
    model.add(keras.layers.BatchNormalization())
    model.add(QDense(NUM_CLASSES, use_bias=False, kernel_quantizer=ternary()))


model.compile()

config = hls4ml.utils.config_from_keras_model(model, granularity='model')

config['Model']['Precision'] = 'ap_fixed<4, 2>'
config['Model']['Strategy'] = 'Resource'
config['Model']['ConvImplementation'] = 'LineBuffer'
config['Model']['ReuseFactor'] = 4096

hls_model = hls4ml.converters.convert_from_keras_model(model,
        hls_config=config,
        output_dir = 'hls4mlprj_cnv2w2a_qkeras', 
        part='xcu250-figd2104-2L-e')
hls_model.compile()
hls_model.build()
