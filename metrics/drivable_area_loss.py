from metrics.metric import Metric
from typing import Dict
import torch



class DrivablelLoss(Metric):
    """
    Pixel-wise loss calculated on heatmap punsihing the predictions on non-drivable area, for cars only
    """

    def __init__(self, args: Dict = None):
        """
        Initialize focal loss
        """
        self.name = 'drivable_loss'
        

    def compute(self, predictions: Dict) -> torch.Tensor:
        """
        Compute drivable loss
        :param predictions: Dictionary with 'pred': predicted heatmap with drivable mask 
        :return:
        """

        # Unpack arguments
        pred = predictions['pred']
        mask = predictions['mask'].view(-1,pred.shape[-2],pred.shape[-1]).unsqueeze(1)
        
        non_drivable_area_mask=~mask

        return  torch.sum(
                non_drivable_area_mask*pred
        )


