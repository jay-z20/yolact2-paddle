# -*- coding: utf-8 -*-
from matplotlib.pyplot import axis
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
#from torch._C import dtype
#from torch.autograd import Variable
from ..box_utils import match, log_sum_exp, decode, center_size, crop, elemwise_mask_iou, elemwise_box_iou, sanitize_coordinates

from data import cfg, mask_type, activation_func

class MultiBoxLoss(nn.Layer):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, pos_threshold, neg_threshold, negpos_ratio):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.negpos_ratio = negpos_ratio

        # If you output a proto mask with this area, your l1 loss will be l1_alpha
        # Note that the area is relative (so 1 would be the entire image)
        self.l1_expected_area = 20*20/70/70
        self.l1_alpha = 0.1

        if cfg.use_class_balanced_conf:
            self.class_instances = None
            self.total_instances = 0

    def forward(self, net, predictions, targets, masks, num_crowds):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            mask preds, and prior boxes from SSD net.
                loc shape: torch.size(batch_size,num_priors,4)
                conf shape: torch.size(batch_size,num_priors,num_classes)
                masks shape: torch.size(batch_size,num_priors,mask_dim)
                priors shape: torch.size(num_priors,4)
                proto* shape: torch.size(batch_size,mask_h,mask_w,mask_dim)

            targets (list<tensor>): Ground truth boxes and labels for a batch,
                shape: [batch_size][num_objs,5] (last idx is the label).

            masks (list<tensor>): Ground truth masks for each object in each image,
                shape: [batch_size][num_objs,im_height,im_width]

            num_crowds (list<int>): Number of crowd annotations per batch. The crowd
                annotations should be the last num_crowds elements of targets and masks.
            
            * Only if mask_type == lincomb
        """

        loc_data  = predictions['loc']
        conf_data = predictions['conf']
        mask_data = predictions['mask']
        priors    = predictions['priors']

        if cfg.mask_type == mask_type.lincomb:
            proto_data = predictions['proto']

        score_data = predictions['score'] if cfg.use_mask_scoring   else None   
        inst_data  = predictions['inst']  if cfg.use_instance_coeff else None
        
        labels = [None] * len(targets) # Used in sem segm loss

        batch_size = loc_data.shape[0]
        num_priors = priors.shape[0]
        num_classes = self.num_classes

        # Match priors (default boxes) and ground truth boxes
        # These tensors will be created with the same device as loc_data
        loc_t = paddle.zeros((batch_size, num_priors, 4))   ################
        gt_box_t = paddle.zeros((batch_size, num_priors, 4))
        conf_t = paddle.zeros((batch_size, num_priors),dtype="int64")
        idx_t = paddle.zeros((batch_size, num_priors),dtype="int64")

        if cfg.use_class_existence_loss:
            class_existence_t = loc_data.new(batch_size, num_classes-1)

        for idx in range(batch_size):
            truths      = targets[idx][:, :-1].detach()   ###############
            labels[idx] = targets[idx][:, -1].detach().astype("int64")

            if cfg.use_class_existence_loss:
                # Construct a one-hot vector for each object and collapse it into an existence vector with max
                # Also it's fine to include the crowd annotations here
                class_existence_t[idx, :] = paddle.eye(num_classes-1)[labels[idx]].max(axis=0)[0]

            # Split the crowd annotations because they come bundled in
            cur_crowds = int(num_crowds[idx])
            if cur_crowds > 0:
                split = lambda x: (x[-cur_crowds:], x[:-cur_crowds])
                crowd_boxes = truths[-cur_crowds:]
                truths = truths[:-cur_crowds] #split(truths)

                # We don't use the crowd labels or masks
                _, labels[idx] = split(labels[idx])
                _, masks[idx]  = split(masks[idx])
            else:
                crowd_boxes = None

            
            match(self.pos_threshold, self.neg_threshold,
                  truths, priors.detach(), labels[idx], crowd_boxes,
                  loc_t, conf_t, idx_t, idx, loc_data[idx])
                  
            gt_box_t[idx, :, :] = paddle.gather(truths,idx_t[idx],0) # truths[idx_t[idx]]

        # wrap targets 
        loc_t.stop_gradient=True
        conf_t.stop_gradient=True
        idx_t.stop_gradient=True

        pos = conf_t > 0
        num_pos = (pos.astype("int")).sum(axis=1, keepdim=True)
        
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        
        losses = {}

        # Localization Loss (Smooth L1)
        if cfg.train_boxes:
            loc_p = paddle.gather_nd(loc_data,paddle.nonzero(pos_idx)).reshape((-1, 4))
            loc_t = paddle.gather_nd(loc_t,paddle.nonzero(pos_idx)).reshape((-1, 4))

            #loc_p = paddle.masked_select(loc_data,pos_idx).reshape((-1, 4)) # loc_data[pos_idx].reshape((-1, 4)) # paddle.masked_select(loc_data,pos_idx)
            #loc_t = paddle.masked_select(loc_t,pos_idx).reshape((-1, 4)) #loc_t[pos_idx].reshape((-1, 4))
            losses['B'] = F.smooth_l1_loss(loc_p, loc_t, reduction='sum') * cfg.bbox_alpha

        if cfg.train_masks:
            if cfg.mask_type == mask_type.direct:
                if cfg.use_gt_bboxes:
                    pos_masks = []
                    for idx in range(batch_size):
                        pos_masks.append(masks[idx][idx_t[idx, pos[idx]]])
                    masks_t = paddle.concat(pos_masks, 0)
                    masks_p = mask_data[pos, :].reshape((-1, cfg.mask_dim))
                    losses['M'] = F.binary_cross_entropy(paddle.clip(masks_p, 0, 1), masks_t, reduction='sum') * cfg.mask_alpha
                else:
                    losses['M'] = self.direct_mask_loss(pos_idx, idx_t, loc_data, mask_data, priors, masks)
            elif cfg.mask_type == mask_type.lincomb:
                ret = self.lincomb_mask_loss(pos, idx_t, loc_data, mask_data, priors, proto_data, masks, gt_box_t, score_data, inst_data, labels)
                if cfg.use_maskiou:
                    loss, maskiou_targets = ret
                else:
                    loss = ret
                losses.update(loss)

                if cfg.mask_proto_loss is not None:
                    if cfg.mask_proto_loss == 'l1':
                        losses['P'] = paddle.mean(paddle.abs(proto_data)) / self.l1_expected_area * self.l1_alpha
                    elif cfg.mask_proto_loss == 'disj':
                        losses['P'] = -paddle.mean(paddle.max(F.log_softmax(proto_data, axis=-1), axis=-1)[0])

        # Confidence loss
        if cfg.use_focal_loss:
            if cfg.use_sigmoid_focal_loss:
                losses['C'] = self.focal_conf_sigmoid_loss(conf_data, conf_t)
            elif cfg.use_objectness_score:
                losses['C'] = self.focal_conf_objectness_loss(conf_data, conf_t)
            else:
                losses['C'] = self.focal_conf_loss(conf_data, conf_t)
        else:
            if cfg.use_objectness_score:
                losses['C'] = self.conf_objectness_loss(conf_data, conf_t, batch_size, loc_p, loc_t, priors)
            else:
                losses['C'] = self.ohem_conf_loss(conf_data, conf_t, pos, batch_size)

        # Mask IoU Loss
        if cfg.use_maskiou and maskiou_targets is not None:
            losses['I'] = self.mask_iou_loss(net, maskiou_targets)

        # These losses also don't depend on anchors
        if cfg.use_class_existence_loss:
            losses['E'] = self.class_existence_loss(predictions['classes'], class_existence_t)
        if cfg.use_semantic_segmentation_loss:
            losses['S'] = self.semantic_segmentation_loss(predictions['segm'], masks, labels)

        # Divide all losses by the number of positives.
        # Don't do it for loss[P] because that doesn't depend on the anchors.
        total_num_pos = num_pos.sum().astype("float32")
        for k in losses:
            if k not in ('P', 'E', 'S'):
                losses[k] /= total_num_pos
            else:
                losses[k] /= batch_size

        # Loss Key:
        #  - B: Box Localization Loss
        #  - C: Class Confidence Loss
        #  - M: Mask Loss
        #  - P: Prototype Loss
        #  - D: Coefficient Diversity Loss
        #  - E: Class Existence Loss
        #  - S: Semantic Segmentation Loss
        return losses

    def class_existence_loss(self, class_data, class_existence_t):
        return cfg.class_existence_alpha * F.binary_cross_entropy_with_logits(class_data, class_existence_t, reduction='sum')

    def semantic_segmentation_loss(self, segment_data, mask_t, class_t, interpolation_mode='bilinear'):
        # Note num_classes here is without the background class so cfg.num_classes-1
        batch_size, num_classes, mask_h, mask_w = segment_data.shape
        loss_s = 0
        
        for idx in range(batch_size):
            cur_segment = segment_data[idx]
            cur_class_t = class_t[idx]

            with paddle.no_grad():
                downsampled_masks = F.interpolate(mask_t[idx].unsqueeze(0), (mask_h, mask_w),
                                                  mode=interpolation_mode, align_corners=False).squeeze(0)
                downsampled_masks = (downsampled_masks>0.5).astype("float32")
                
                # Construct Semantic Segmentation
                segment_t = paddle.zeros_like(cur_segment)
                segment_t.stop_gradient=True
                for obj_idx in range(downsampled_masks.shape[0]):
                    segment_t[cur_class_t[obj_idx]] = paddle.maximum(segment_t[cur_class_t[obj_idx]], downsampled_masks[obj_idx])
            
            loss_s += F.binary_cross_entropy_with_logits(cur_segment, segment_t, reduction='sum')
        
        return loss_s / mask_h / mask_w * cfg.semantic_segmentation_alpha


    def ohem_conf_loss(self, conf_data, conf_t, pos, num):
        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.reshape((-1, self.num_classes))
        if cfg.ohem_use_most_confident:
            # i.e. max(softmax) along classes > 0 
            batch_conf = F.softmax(batch_conf, axis=1)
            loss_c, _ = batch_conf[:, 1:].max(axis=1)
        else:
            # i.e. -softmax(class 0 confidence)
            loss_c = log_sum_exp(batch_conf) - batch_conf[:, 0]

        # Hard Negative Mining
        loss_c = loss_c.reshape((num, -1))
        loss_c = paddle.where(pos,
                                paddle.zeros_like(loss_c), loss_c)
        
        loss_c = paddle.where(conf_t < 0,
                                paddle.zeros_like(loss_c), loss_c)
        #loss_c[pos]        = 0 # filter out pos boxes
        #loss_c[conf_t < 0] = 0 # filter out neutrals (conf_t = -1)
        loss_idx = loss_c.argsort(1, descending=True)
        idx_rank = loss_idx.argsort(1)
        num_pos = pos.astype("int64").sum(1, keepdim=True)
        num_neg = paddle.clip(self.negpos_ratio*num_pos, max=pos.shape[1]-1)
        neg = (idx_rank < num_neg.expand_as(idx_rank)).astype("int32")
        
        # Just in case there aren't enough negatives, don't start using positives as negatives
        neg = paddle.where(pos,
                                paddle.zeros_like(neg), neg)
        
        neg = paddle.where(conf_t < 0,
                                paddle.zeros_like(neg), neg)
        neg = paddle.cast(neg, 'bool')
        #neg[pos]        = 0
        #neg[conf_t < 0] = 0 # Filter out neutrals

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        all_idx = paddle.logical_or(pos_idx, neg_idx)
        conf_p = paddle.gather_nd(conf_data,paddle.nonzero(all_idx)).reshape((-1, self.num_classes))
        all_p = paddle.logical_or(pos, neg)
        #targets_weighted = paddle.masked_select(conf_t,all_p)
        targets_weighted = paddle.gather_nd(conf_t,paddle.nonzero(all_p))

        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='none')

        if cfg.use_class_balanced_conf:
            # Lazy initialization
            if self.class_instances is None:
                self.class_instances = paddle.zeros(self.num_classes)
            
            classes, counts = targets_weighted.unique(return_counts=True)
            
            for _cls, _cnt in zip(classes.cpu().numpy(), counts.cpu().numpy()):
                self.class_instances[_cls] += _cnt

            self.total_instances += targets_weighted.shape[0]

            weighting = 1 - (self.class_instances[targets_weighted] / self.total_instances)
            weighting = paddle.clip(weighting, min=1/self.num_classes)

            # If you do the math, the average weight of self.class_instances is this
            avg_weight = (self.num_classes - 1) / self.num_classes

            loss_c = (loss_c * weighting).sum() / avg_weight
        else:
            loss_c = loss_c.sum()
        
        return cfg.conf_alpha * loss_c

    def focal_conf_loss(self, conf_data, conf_t):
        """
        Focal loss as described in https://arxiv.org/pdf/1708.02002.pdf
        Adapted from https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
        Note that this uses softmax and not the original sigmoid from the paper.
        """
        conf_t = conf_t.reshape((-1)) # [batch_size*num_priors]
        conf_data = conf_data.reshape((-1, conf_data.shape[-1])) # [batch_size*num_priors, num_classes]

        # Ignore neutral samples (class < 0)
        keep = (conf_t >= 0).astype("float32")
        conf_t[conf_t < 0] = 0 # so that gather doesn't drum up a fuss

        logpt = F.log_softmax(conf_data, axis=-1)
        logpt = logpt.gather(1, conf_t.unsqueeze(-1))
        logpt = logpt.reshape((-1))
        pt    = logpt.exp()

        # I adapted the alpha_t calculation here from
        # https://github.com/pytorch/pytorch/blob/master/modules/detectron/softmax_focal_loss_op.cu
        # You'd think you want all the alphas to sum to one, but in the original implementation they
        # just give background an alpha of 1-alpha and each forground an alpha of alpha.
        background = (conf_t == 0).astype("float32")
        at = (1 - cfg.focal_loss_alpha) * background + cfg.focal_loss_alpha * (1 - background)

        loss = -at * (1 - pt) ** cfg.focal_loss_gamma * logpt

        # See comment above for keep
        return cfg.conf_alpha * (loss * keep).sum()
    
    def focal_conf_sigmoid_loss(self, conf_data, conf_t):
        """
        Focal loss but using sigmoid like the original paper.
        Note: To make things mesh easier, the network still predicts 81 class confidences in this mode.
              Because retinanet originally only predicts 80, we simply just don't use conf_data[..., 0]
        """
        num_classes = conf_data.shape[-1]

        conf_t = conf_t.reshape((-1)) # [batch_size*num_priors]
        conf_data = conf_data.reshape((-1, num_classes)) # [batch_size*num_priors, num_classes]

        # Ignore neutral samples (class < 0)
        keep = (conf_t >= 0).astype("float32")
        conf_t[conf_t < 0] = 0 # can't mask with -1, so filter that out

        # Compute a one-hot embedding of conf_t
        # From https://github.com/kuangliu/pytorch-retinanet/blob/master/utils.py
        conf_one_t = paddle.eye(num_classes)[conf_t]
        conf_pm_t  = conf_one_t * 2 - 1 # -1 if background, +1 if forground for specific class

        logpt = F.logsigmoid(conf_data * conf_pm_t) # note: 1 - sigmoid(x) = sigmoid(-x)
        pt    = logpt.exp()

        at = cfg.focal_loss_alpha * conf_one_t + (1 - cfg.focal_loss_alpha) * (1 - conf_one_t)
        at[..., 0] = 0 # Set alpha for the background class to 0 because sigmoid focal loss doesn't use it

        loss = -at * (1 - pt) ** cfg.focal_loss_gamma * logpt
        loss = keep * loss.sum(axis=-1)

        return cfg.conf_alpha * loss.sum()
    
    def focal_conf_objectness_loss(self, conf_data, conf_t):
        """
        Instead of using softmax, use class[0] to be the objectness score and do sigmoid focal loss on that.
        Then for the rest of the classes, softmax them and apply CE for only the positive examples.

        If class[0] = 1 implies forground and class[0] = 0 implies background then you achieve something
        similar during test-time to softmax by setting class[1:] = softmax(class[1:]) * class[0] and invert class[0].
        """

        conf_t = conf_t.reshape((-1)) # [batch_size*num_priors]
        conf_data = conf_data.reshape((-1, conf_data.shape[-1])) # [batch_size*num_priors, num_classes]

        # Ignore neutral samples (class < 0)
        keep = (conf_t >= 0).astype("float32")
        conf_t[conf_t < 0] = 0 # so that gather doesn't drum up a fuss

        background = (conf_t == 0).astype("float32")
        at = (1 - cfg.focal_loss_alpha) * background + cfg.focal_loss_alpha * (1 - background)

        logpt = F.logsigmoid(conf_data[:, 0]) * (1 - background) + F.logsigmoid(-conf_data[:, 0]) * background
        pt    = logpt.exp()

        obj_loss = -at * (1 - pt) ** cfg.focal_loss_gamma * logpt

        # All that was the objectiveness loss--now time for the class confidence loss
        pos_mask = conf_t > 0
        conf_data_pos = (conf_data[:, 1:])[pos_mask] # Now this has just 80 classes
        conf_t_pos    = conf_t[pos_mask] - 1         # So subtract 1 here

        class_loss = F.cross_entropy(conf_data_pos, conf_t_pos, reduction='sum')

        return cfg.conf_alpha * (class_loss + (obj_loss * keep).sum())
    
    def conf_objectness_loss(self, conf_data, conf_t, batch_size, loc_p, loc_t, priors):
        """
        Instead of using softmax, use class[0] to be p(obj) * p(IoU) as in YOLO.
        Then for the rest of the classes, softmax them and apply CE for only the positive examples.
        """

        conf_t = conf_t.reshape((-1)) # [batch_size*num_priors]
        conf_data = conf_data.reshape((-1, conf_data.shape[-1])) # [batch_size*num_priors, num_classes]

        pos_mask = (conf_t > 0)
        neg_mask = (conf_t == 0)

        obj_data = conf_data[:, 0]
        obj_data_pos = obj_data[pos_mask]
        obj_data_neg = obj_data[neg_mask]

        # Don't be confused, this is just binary cross entropy similified
        obj_neg_loss = - F.logsigmoid(-obj_data_neg).sum()

        with paddle.no_grad():
            pos_priors = priors.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 4)[pos_mask, :]

            boxes_pred = decode(loc_p, pos_priors, cfg.use_yolo_regressors)
            boxes_targ = decode(loc_t, pos_priors, cfg.use_yolo_regressors)

            iou_targets = elemwise_box_iou(boxes_pred, boxes_targ)

        obj_pos_loss = - iou_targets * F.logsigmoid(obj_data_pos) - (1 - iou_targets) * F.logsigmoid(-obj_data_pos)
        obj_pos_loss = obj_pos_loss.sum()

        # All that was the objectiveness loss--now time for the class confidence loss
        conf_data_pos = (conf_data[:, 1:])[pos_mask] # Now this has just 80 classes
        conf_t_pos    = conf_t[pos_mask] - 1         # So subtract 1 here

        class_loss = F.cross_entropy(conf_data_pos, conf_t_pos, reduction='sum')

        return cfg.conf_alpha * (class_loss + obj_pos_loss + obj_neg_loss)


    def direct_mask_loss(self, pos_idx, idx_t, loc_data, mask_data, priors, masks):
        """ Crops the gt masks using the predicted bboxes, scales them down, and outputs the BCE loss. """
        loss_m = 0
        for idx in range(mask_data.shape[0]):
            with paddle.no_grad():
                cur_pos_idx = pos_idx[idx, :, :]
                cur_pos_idx_squeezed = cur_pos_idx[:, 1]

                # Shape: [num_priors, 4], decoded predicted bboxes
                pos_bboxes = decode(loc_data[idx, :, :], priors.data, cfg.use_yolo_regressors)
                pos_bboxes = pos_bboxes[cur_pos_idx].reshape((-1, 4)).clip(0, 1)
                pos_lookup = idx_t[idx, cur_pos_idx_squeezed]

                cur_masks = masks[idx]
                pos_masks = cur_masks[pos_lookup, :, :]
                
                # Convert bboxes to absolute coordinates
                num_pos, img_height, img_width = pos_masks.shape

                # Take care of all the bad behavior that can be caused by out of bounds coordinates
                x1, x2 = sanitize_coordinates(pos_bboxes[:, 0], pos_bboxes[:, 2], img_width)
                y1, y2 = sanitize_coordinates(pos_bboxes[:, 1], pos_bboxes[:, 3], img_height)

                # Crop each gt mask with the predicted bbox and rescale to the predicted mask size
                # Note that each bounding box crop is a different size so I don't think we can vectorize this
                scaled_masks = []
                for jdx in range(num_pos):
                    tmp_mask = pos_masks[jdx, y1[jdx]:y2[jdx], x1[jdx]:x2[jdx]]

                    # Restore any dimensions we've left out because our bbox was 1px wide
                    while tmp_mask.dim() < 2:
                        tmp_mask = tmp_mask.unsqueeze(0)

                    new_mask = F.adaptive_avg_pool2d(tmp_mask.unsqueeze(0), cfg.mask_size)
                    scaled_masks.append(new_mask.reshape((1, -1)))

                mask_t = paddle.concat(scaled_masks, 0).gt(0.5).astype("float32") # Threshold downsampled mask
            
            pos_mask_data = mask_data[idx, cur_pos_idx_squeezed, :]
            loss_m += F.binary_cross_entropy(paddle.clip(pos_mask_data, 0, 1), mask_t, reduction='sum') * cfg.mask_alpha

        return loss_m
    

    def coeff_diversity_loss(self, coeffs, instance_t):
        """
        coeffs     should be size [num_pos, num_coeffs]
        instance_t should be size [num_pos] and be values from 0 to num_instances-1
        """
        num_pos = coeffs.shape[0]
        instance_t = instance_t.reshape((-1)) # juuuust to make sure

        coeffs_norm = F.normalize(coeffs, axis=1)
        cos_sim = paddle.mm(coeffs_norm, coeffs_norm.t())

        inst_eq = (instance_t[:, None].expand_as(cos_sim) == instance_t[None, :].expand_as(cos_sim)).float()

        # Rescale to be between 0 and 1
        cos_sim = (cos_sim + 1) / 2

        # If they're the same instance, use cosine distance, else use cosine similarity
        loss = (1 - cos_sim) * inst_eq + cos_sim * (1 - inst_eq)

        # Only divide by num_pos once because we're summing over a num_pos x num_pos tensor
        # and all the losses will be divided by num_pos at the end, so just one extra time.
        return cfg.mask_proto_coeff_diversity_alpha * loss.sum() / num_pos


    def lincomb_mask_loss(self, pos, idx_t, loc_data, mask_data, priors, proto_data, masks, gt_box_t, score_data, inst_data, labels, interpolation_mode='bilinear'):
        mask_h = proto_data.shape[1]
        mask_w = proto_data.shape[2]

        process_gt_bboxes = cfg.mask_proto_normalize_emulate_roi_pooling or cfg.mask_proto_crop

        if cfg.mask_proto_remove_empty_masks:
            # Make sure to store a copy of this because we edit it to get rid of all-zero masks
            pos = pos.clone()

        loss_m = 0
        loss_d = 0 # Coefficient diversity loss

        maskiou_t_list = []
        maskiou_net_input_list = []
        label_t_list = []

        for idx in range(mask_data.shape[0]):
            with paddle.no_grad():
                downsampled_masks = F.interpolate(masks[idx].unsqueeze(0), (mask_h, mask_w),
                                                  mode=interpolation_mode, align_corners=False).squeeze(0)
                downsampled_masks = downsampled_masks.transpose((1, 2, 0))

                if cfg.mask_proto_binarize_downsampled_gt:
                    downsampled_masks = (downsampled_masks>0.5).astype("float32") #downsampled_masks.gt(0.5).astype("float32")

                if cfg.mask_proto_remove_empty_masks:
                    # Get rid of gt masks that are so small they get downsampled away
                    very_small_masks = (downsampled_masks.sum(axis=(0,1)) <= 0.0001)
                    for i in range(very_small_masks.shape[0]):
                        if very_small_masks[i]:
                            pos[idx, idx_t[idx] == i] = 0

                if cfg.mask_proto_reweight_mask_loss:
                    # Ensure that the gt is binary
                    if not cfg.mask_proto_binarize_downsampled_gt:
                        bin_gt = (downsampled_masks>0.5).astype("float32") # downsampled_masks.gt(0.5).astype("float32")
                    else:
                        bin_gt = downsampled_masks

                    gt_foreground_norm = bin_gt     / (paddle.sum(bin_gt,   axis=(0,1), keepdim=True) + 0.0001)
                    gt_background_norm = (1-bin_gt) / (paddle.sum(1-bin_gt, axis=(0,1), keepdim=True) + 0.0001)

                    mask_reweighting   = gt_foreground_norm * cfg.mask_proto_reweight_coeff + gt_background_norm
                    mask_reweighting  *= mask_h * mask_w

            cur_pos = pos.astype("int")[idx] #pos[idx]
            #pos_idx_t = paddle.masked_select(idx_t[idx],cur_pos.astype("bool")) #idx_t[idx, cur_pos]
            pos_idx_t = paddle.gather_nd(idx_t[idx],paddle.nonzero(cur_pos))
            if process_gt_bboxes:
                # Note: this is in point-form
                if cfg.mask_proto_crop_with_pred_box:
                    #pos_gt_box_t = paddle.masked_select(decode(loc_data[idx, :, :], priors.data, cfg.use_yolo_regressors),cur_pos.astype("bool"))
                    pos_gt_box_t = paddle.gather_nd(decode(loc_data[idx, :, :], priors.data, cfg.use_yolo_regressors),paddle.nonzero(cur_pos))
                else:
                    pos_gt_box_t = paddle.gather(gt_box_t[idx],paddle.nonzero(cur_pos)) #gt_box_t[idx, cur_pos]

            if pos_idx_t.shape[0] == 0:
                continue

            proto_masks = proto_data[idx]
            proto_coef  = paddle.gather(mask_data[idx],paddle.nonzero(cur_pos)) #mask_data[idx, cur_pos, :] # 
            if cfg.use_mask_scoring:
                mask_scores = paddle.gather(score_data[idx],paddle.nonzero(cur_pos)) #score_data[idx, cur_pos, :]

            if cfg.mask_proto_coeff_diversity_loss:
                if inst_data is not None:
                    div_coeffs = paddle.gather(inst_data[idx],paddle.nonzero(cur_pos)) #inst_data[idx, cur_pos, :]
                else:
                    div_coeffs = proto_coef

                loss_d += self.coeff_diversity_loss(div_coeffs, pos_idx_t)
            
            # If we have over the allowed number of masks, select a random sample
            old_num_pos = proto_coef.shape[0]
            if old_num_pos > cfg.masks_to_train:
                perm = paddle.randperm(proto_coef.shape[0])
                select = perm[:cfg.masks_to_train]

                proto_coef = paddle.gather(proto_coef,select,0) #proto_coef[select, :]
                pos_idx_t  = paddle.gather(pos_idx_t,select)  #pos_idx_t[select]
                
                if process_gt_bboxes:
                    pos_gt_box_t = paddle.gather(pos_gt_box_t,select,0) #pos_gt_box_t[select, :]
                if cfg.use_mask_scoring:
                    mask_scores = paddle.gather(mask_scores,select,0) #mask_scores[select, :]

            num_pos = proto_coef.shape[0]
            mask_t = paddle.gather(downsampled_masks,pos_idx_t,2)
            #mask_t = downsampled_masks[:, :, pos_idx_t]    
            label_t = paddle.gather(labels[idx],pos_idx_t)
            #label_t = labels[idx][pos_idx_t]     

            # Size: [mask_h, mask_w, num_pos]
            pred_masks = paddle.mm(proto_masks,proto_coef.t())
            pred_masks = cfg.mask_proto_mask_activation(pred_masks)

            if cfg.mask_proto_double_loss:
                if cfg.mask_proto_mask_activation == activation_func.sigmoid:
                    pre_loss = F.binary_cross_entropy(paddle.clip(pred_masks, 0, 1), mask_t, reduction='sum')
                else:
                    pre_loss = F.smooth_l1_loss(pred_masks, mask_t, reduction='sum')
                
                loss_m += cfg.mask_proto_double_loss_alpha * pre_loss

            if cfg.mask_proto_crop:
                pred_masks = crop(pred_masks, pos_gt_box_t)
            
            if cfg.mask_proto_mask_activation == activation_func.sigmoid:
                pre_loss = F.binary_cross_entropy(paddle.clip(pred_masks, 0, 1), mask_t, reduction='none')
            else:
                pre_loss = F.smooth_l1_loss(pred_masks, mask_t, reduction='none')

            if cfg.mask_proto_normalize_mask_loss_by_sqrt_area:
                gt_area  = paddle.sum(mask_t, axis=(0, 1), keepdim=True)
                pre_loss = pre_loss / (paddle.sqrt(gt_area) + 0.0001)
            
            if cfg.mask_proto_reweight_mask_loss:
                pre_loss = pre_loss * mask_reweighting[:, :, pos_idx_t]
                
            if cfg.mask_proto_normalize_emulate_roi_pooling:
                weight = mask_h * mask_w if cfg.mask_proto_crop else 1
                pos_gt_csize = center_size(pos_gt_box_t)
                gt_box_width  = pos_gt_csize[:, 2] * mask_w
                gt_box_height = pos_gt_csize[:, 3] * mask_h
                pre_loss = pre_loss.sum(axis=(0, 1)) / gt_box_width / gt_box_height * weight

            # If the number of masks were limited scale the loss accordingly
            if old_num_pos > num_pos:
                pre_loss *= old_num_pos / num_pos

            loss_m += paddle.sum(pre_loss)

            if cfg.use_maskiou:
                if cfg.discard_mask_area > 0:
                    gt_mask_area = paddle.sum(mask_t, axis=(0, 1))
                    select = gt_mask_area > cfg.discard_mask_area

                    if paddle.sum(select.astype("int")) < 1:
                        continue

                    pos_gt_box_t = paddle.gather(pos_gt_box_t,paddle.nonzero(select)) 
                    pred_masks = paddle.gather(pred_masks,paddle.nonzero(select),axis=2) 
                    mask_t = paddle.gather(mask_t,paddle.nonzero(select),axis=2) 
                    label_t = paddle.gather(label_t,paddle.nonzero(select)) 

                maskiou_net_input = pred_masks.transpose((2, 0, 1)).unsqueeze(1)
                pred_masks = (pred_masks>0.5).astype("float32")               
                maskiou_t = self._mask_iou(pred_masks, mask_t)
                
                maskiou_net_input_list.append(maskiou_net_input)
                maskiou_t_list.append(maskiou_t)
                label_t_list.append(label_t)
        
        losses = {'M': loss_m * cfg.mask_alpha / mask_h / mask_w}
        
        if cfg.mask_proto_coeff_diversity_loss:
            losses['D'] = loss_d

        if cfg.use_maskiou:
            # discard_mask_area discarded every mask in the batch, so nothing to do here
            if len(maskiou_t_list) == 0:
                return losses, None

            maskiou_t = paddle.concat(maskiou_t_list)
            label_t = paddle.concat(label_t_list)
            maskiou_net_input = paddle.concat(maskiou_net_input_list)

            num_samples = maskiou_t.shape[0]
            if cfg.maskious_to_train > 0 and num_samples > cfg.maskious_to_train:
                perm = paddle.randperm(num_samples)
                select = perm[:cfg.masks_to_train]
                maskiou_t = maskiou_t[select]
                label_t = label_t[select]
                maskiou_net_input = maskiou_net_input[select]

            return losses, [maskiou_net_input, maskiou_t, label_t]

        return losses

    def _mask_iou(self, mask1, mask2):
        intersection = paddle.sum(mask1*mask2, axis=(0, 1))
        area1 = paddle.sum(mask1, axis=(0, 1))
        area2 = paddle.sum(mask2, axis=(0, 1))
        union = (area1 + area2) - intersection
        ret = intersection / union
        return ret

    def mask_iou_loss(self, net, maskiou_targets):
        maskiou_net_input, maskiou_t, label_t = maskiou_targets

        maskiou_p = net.maskiou_net(maskiou_net_input)

        label_t = label_t.unsqueeze(1) #[:, None]
        maskiou_p = paddle.gather(maskiou_p, axis=1, index=label_t)[:,0]

        loss_i = F.smooth_l1_loss(maskiou_p, maskiou_t, reduction='sum')
        
        return loss_i * cfg.maskiou_alpha
