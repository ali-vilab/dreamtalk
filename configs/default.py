from yacs.config import CfgNode as CN


_C = CN()
_C.TAG = "style_id_emotion"
_C.DECODER_TYPE = "DisentangleDecoder"
_C.CONTENT_ENCODER_TYPE = "ContentW2VEncoder"
_C.STYLE_ENCODER_TYPE = "StyleEncoder"

_C.DIFFNET_TYPE = "DiffusionNet"

_C.WIN_SIZE = 5
_C.D_MODEL = 256

_C.DATASET = CN()
_C.DATASET.FACE3D_DIM = 64
_C.DATASET.NUM_FRAMES = 64
_C.DATASET.STYLE_MAX_LEN = 256

_C.TRAIN = CN()
_C.TRAIN.FACE3D_LATENT = CN()
_C.TRAIN.FACE3D_LATENT.TYPE = "face3d"

_C.DIFFUSION = CN()
_C.DIFFUSION.PREDICT_WHAT = "x0"  # noise | x0
_C.DIFFUSION.SCHEDULE = CN()
_C.DIFFUSION.SCHEDULE.NUM_STEPS = 1000
_C.DIFFUSION.SCHEDULE.BETA_1 = 1e-4
_C.DIFFUSION.SCHEDULE.BETA_T = 0.02
_C.DIFFUSION.SCHEDULE.MODE = "linear"

_C.CONTENT_ENCODER = CN()
_C.CONTENT_ENCODER.d_model = _C.D_MODEL
_C.CONTENT_ENCODER.nhead = 8
_C.CONTENT_ENCODER.num_encoder_layers = 3
_C.CONTENT_ENCODER.dim_feedforward = 4 * _C.D_MODEL
_C.CONTENT_ENCODER.dropout = 0.1
_C.CONTENT_ENCODER.activation = "relu"
_C.CONTENT_ENCODER.normalize_before = False
_C.CONTENT_ENCODER.pos_embed_len = 2 * _C.WIN_SIZE + 1

_C.STYLE_ENCODER = CN()
_C.STYLE_ENCODER.d_model = _C.D_MODEL
_C.STYLE_ENCODER.nhead = 8
_C.STYLE_ENCODER.num_encoder_layers = 3
_C.STYLE_ENCODER.dim_feedforward = 4 * _C.D_MODEL
_C.STYLE_ENCODER.dropout = 0.1
_C.STYLE_ENCODER.activation = "relu"
_C.STYLE_ENCODER.normalize_before = False
_C.STYLE_ENCODER.pos_embed_len = _C.DATASET.STYLE_MAX_LEN
_C.STYLE_ENCODER.aggregate_method = (
    "self_attention_pooling"  # average | self_attention_pooling
)
# _C.STYLE_ENCODER.input_dim = _C.DATASET.FACE3D_DIM

_C.DECODER = CN()
_C.DECODER.d_model = _C.D_MODEL
_C.DECODER.nhead = 8
_C.DECODER.num_decoder_layers = 3
_C.DECODER.dim_feedforward = 4 * _C.D_MODEL
_C.DECODER.dropout = 0.1
_C.DECODER.activation = "relu"
_C.DECODER.normalize_before = False
_C.DECODER.return_intermediate_dec = False
_C.DECODER.pos_embed_len = 2 * _C.WIN_SIZE + 1
_C.DECODER.network_type = "TransformerDecoder"
_C.DECODER.dynamic_K = None
_C.DECODER.dynamic_ratio = None
# _C.DECODER.output_dim = _C.DATASET.FACE3D_DIM
# LSFM basis:
# _C.DECODER.upper_face3d_indices = tuple(list(range(19)) + list(range(46, 51)))
# _C.DECODER.lower_face3d_indices = tuple(range(19, 46))
# BFM basis:
# fmt: off
_C.DECODER.upper_face3d_indices = [6, 8, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] 
# fmt: on
_C.DECODER.lower_face3d_indices = [0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14]

_C.CF_GUIDANCE = CN()
_C.CF_GUIDANCE.TRAINING = True
_C.CF_GUIDANCE.INFERENCE = True
_C.CF_GUIDANCE.NULL_PROB = 0.1
_C.CF_GUIDANCE.SCALE = 1.0

_C.INFERENCE = CN()
_C.INFERENCE.CHECKPOINT = "checkpoints/denoising_network.pth"


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    return _C.clone()
