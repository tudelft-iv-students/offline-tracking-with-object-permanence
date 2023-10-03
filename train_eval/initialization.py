# Import datasets
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from datasets.nuScenes.prediction import PredictHelper_occ
from datasets.interface import TrajectoryDataset
from datasets.nuScenes.nuScenes_raster import NuScenesRaster
from datasets.nuScenes.nuScenes_vector import NuScenesVector
from datasets.nuScenes.nuScenes_graphs import NuScenesGraphs
from datasets.nuScenes.nuScenes_graphs_occ_v2 import NuScenesGraphs_OCC
from datasets.nuScenes.nuScenes_graphs_match import NuScenesGraphs_MATCH

# Import models
from models.model import PredictionModel

from models.encoders.home_encoder import HomeEncoder
from models.encoders.raster_encoder import RasterEncoder
from models.encoders.polyline_subgraph import PolylineSubgraphs
from models.encoders.pgp_encoder import PGPEncoder
from models.encoders.pgp_encoder_v2 import PGPEncoder_occ
from models.encoders.match_encoder import MatchEncoder
from models.aggregators.concat import Concat
from models.aggregators.global_attention import GlobalAttention
from models.aggregators.goal_conditioned import GoalConditioned
from models.aggregators.home_agg import Sample2DAggregator
from models.aggregators.home_orig import HomeAggregator
from models.aggregators.attention_occ import Attention_occ
from models.aggregators.match_agg import Match_agg
from models.aggregators.pgp import PGP
from models.aggregators.ram_agg import RamAggregator
from models.decoders.mtp import MTP
from models.decoders.multipath import Multipath
from models.decoders.covernet import CoverNet
# from models.decoders.lvm import LVM
from models.decoders.ram_decoder import RamDecoder
from models.decoders.heatmap import HTMAP
from models.decoders.home_decoder import HomeDecoder
from models.decoders.home_atten_decoder import HomeDecoder_attn
from models.decoders.mlp_occ import MLP_occ
from models.decoders.match_decoder import Match_decoder
# Import metrics
from metrics.mtp_loss import MTPLoss
from metrics.min_ade import MinADEK
from metrics.min_ade_l1 import MinL1K
from metrics.min_ade_loss import MinADE_loss
from metrics.min_fde import MinFDEK
from metrics.miss_rate import MissRateK
from metrics.covernet_loss import CoverNetLoss
from metrics.pi_bc import PiBehaviorCloning
from metrics.goal_pred_nll import GoalPredictionNLL
from metrics.focal_loss import FocalLoss
from metrics.l1_loss import L1_loss
from metrics.huber import Huber
from metrics.yaw_loss import Yaw_loss
from metrics.ade import ADE
from metrics.mr import MissRate
from metrics.binary_focal import Binary_focal_loss
from metrics.drivable_area_loss import DrivablelLoss
from metrics.match_accuracy import Match_accuracy
from typing import List, Dict, Union


# Datasets
def initialize_dataset(dataset_type: str, args: List) -> TrajectoryDataset:
    """
    Helper function to initialize appropriate dataset by dataset type string
    """
    # TODO: Add more datasets as implemented
    dataset_classes = {'nuScenes_single_agent_raster': NuScenesRaster,
                       'nuScenes_single_agent_vector': NuScenesVector,
                       'nuScenes_single_agent_graphs': NuScenesGraphs,
                       'nuScenes_single_agent_graphs_occ': NuScenesGraphs_OCC,
                       'nuScenes_single_agent_match': NuScenesGraphs_MATCH
                       }
    return dataset_classes[dataset_type](*args)


def get_specific_args(dataset_name: str, data_root: str, version: str = None, helper_type: str = None) -> List:
    """
    Helper function to get dataset specific arguments.
    """
    # TODO: Add more datasets as implemented
    specific_args = []
    if dataset_name == 'nuScenes':
        ns = NuScenes(version, dataroot=data_root)
        if helper_type=='occ':
            pred_helper = PredictHelper_occ(ns)
        else:
            pred_helper = PredictHelper(ns)
        specific_args.append(pred_helper)

    return specific_args


# Models
def initialize_prediction_model(encoder_type: str, aggregator_type: str, decoder_type: str,
                                encoder_args: Dict, aggregator_args: Union[Dict, None], decoder_args: Dict):
    """
    Helper function to initialize appropriate encoder, aggegator and decoder models
    """
    encoder = initialize_encoder(encoder_type, encoder_args)
    aggregator = initialize_aggregator(aggregator_type, aggregator_args)
    decoder = initialize_decoder(decoder_type, decoder_args)
    model = PredictionModel(encoder, aggregator, decoder)

    return model


def initialize_encoder(encoder_type: str, encoder_args: Dict):
    """
    Initialize appropriate encoder by type.
    """
    # TODO: Update as we add more encoder types
    encoder_mapping = {
        'raster_encoder': RasterEncoder,
        'polyline_subgraphs': PolylineSubgraphs,
        'pgp_encoder': PGPEncoder,
        'pgp_encoder_occ': PGPEncoder_occ,
        'home_encoder':HomeEncoder,
        'match_encoder':MatchEncoder
    }

    return encoder_mapping[encoder_type](encoder_args)


def initialize_aggregator(aggregator_type: str, aggregator_args: Union[Dict, None]):
    """
    Initialize appropriate aggregator by type.
    """
    # TODO: Update as we add more aggregator types
    aggregator_mapping = {
        'concat': Concat,
        'global_attention': GlobalAttention,
        'gc': GoalConditioned,
        'pgp': PGP,
        'attn_occ': Attention_occ,
        'home_agg': HomeAggregator,
        '2D_sample':Sample2DAggregator,
        'ram':RamAggregator,
        'match_agg':Match_agg
    }

    if aggregator_args:
        return aggregator_mapping[aggregator_type](aggregator_args)
    else:
        return aggregator_mapping[aggregator_type]()


def initialize_decoder(decoder_type: str, decoder_args: Dict):
    """
    Initialize appropriate decoder by type.
    """
    # TODO: Update as we add more decoder types
    decoder_mapping = {
        'mtp': MTP,
        'multipath': Multipath,
        'covernet': CoverNet,
        # 'lvm': LVM,
        'heatmap':HTMAP,
        'home':HomeDecoder,
        'ram_decoder':RamDecoder,
        'home_atten':HomeDecoder_attn,
        'mlp_occ':MLP_occ,
        'match_decoder':Match_decoder
    }

    return decoder_mapping[decoder_type](decoder_args)


# Metrics
def initialize_metric(metric_type: str, metric_args: Dict = None):
    """
    Initialize appropriate metric by type.
    """
    # TODO: Update as we add more metrics
    metric_mapping = {
        'mtp_loss': MTPLoss,
        'covernet_loss': CoverNetLoss,
        'min_ade_k': MinADEK,
        'min_ade_loss': MinADE_loss,
        'min_fde_k': MinFDEK,
        'min_l1_k':MinL1K,
        'miss_rate_k': MissRateK,
        'pi_bc': PiBehaviorCloning,
        'goal_pred_nll': GoalPredictionNLL,
        'focal_loss':FocalLoss,
        'drivable_loss':DrivablelLoss,
        'l1_loss': L1_loss,
        'yaw_loss':Yaw_loss,
        'ade':ADE,
        'miss_rate':MissRate,
        'binary_focal_loss':Binary_focal_loss,
        'match_accuracy':Match_accuracy,
        'huber':Huber
    }

    if metric_args is not None:
        return metric_mapping[metric_type](metric_args)
    else:
        return metric_mapping[metric_type]()
