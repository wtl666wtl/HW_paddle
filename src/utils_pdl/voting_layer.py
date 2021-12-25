import paddle
import paddle.nn as nn


class Voting(nn.Layer):
    """
    Voting Layer computes a new row-stotatic matrix with softmax. A large number (alpha) is multiplied to the input
    stochastic matrix to scale up the difference.
    Parameter: value multiplied before softmax alpha
               threshold that will ignore such points while calculating displacement in pixels pixel_thresh
    Input: permutation or doubly stochastic matrix s
           ///point set on source image P_src
           ///point set on target image P_tgt
           ground truth number of effective points in source image ns_gt
    Output: softmax matrix s
    """
    def __init__(self, alpha=200, pixel_thresh=None):
        super(Voting, self).__init__()
        self.alpha = alpha
        self.softmax = nn.Softmax(axis=-1)  # Voting among columns
        self.pixel_thresh = pixel_thresh

    def forward(self, s, nrow_gt, ncol_gt=None):
        ret_s = paddle.zeros_like(s)
        # filter dummy nodes
        for b, n in enumerate(nrow_gt):
            if ncol_gt is None:
                ret_s[b, 0:n, :] = \
                    self.softmax(self.alpha * s[b, 0:n, :])
            else:
                tmp = int(ncol_gt[b].numpy())
                ret_s[b, 0:int(n), 0:tmp] =\
                    self.softmax(self.alpha * s[b, 0:int(n), 0:tmp])

        return ret_s


if __name__ == '__main__':
    import numpy as np
    s = np.array(
                [[[1.78497553, 2.33776569, 2.48464251, 1.87491512, 1.98989379, 2.40511894, 2.37472200, 1.69409716, 1.89110827, 2.51438236, 1.67087138],
         [1.36833346, 1.76622033, 1.82004869, 1.46487260, 1.51084673, 1.78871894, 1.78109431, 1.30949831, 1.40313423, 1.80326819, 1.31016243],
         [1.15162337, 1.33354855, 1.38642883, 1.14695048, 1.20355141, 1.34822941, 1.36444771, 1.13208723, 1.07972169, 1.37298024, 0.99443769],
         [1.31082046, 1.70131564, 1.71726143, 1.38820112, 1.46717584, 1.71423769, 1.72037578, 1.23355246, 1.36826181, 1.77211881, 1.24508083],
         [1.10933423, 1.44153237, 1.55079007, 1.22282863, 1.25531042, 1.48060524, 1.48228788, 1.07130587, 1.26667666, 1.50533378, 1.08315337],
         [1.68768573, 2.01056314, 2.08990502, 1.66733491, 1.71560609, 2.05998874, 2.04883575, 1.54030490, 1.61696517, 2.10970616, 1.50980115],
         [1.86688662, 2.46825933, 2.56624627, 2.01873660, 2.08246160, 2.59437275, 2.51757693, 1.74435234, 1.94333422, 2.56985116, 1.76538563],
         [1.46393406, 1.89939821, 1.95997715, 1.56661391, 1.59947729, 1.99042201, 1.91306901, 1.40414703, 1.49506271, 1.98714066, 1.35258973],
         [1.42765772, 1.91140413, 1.95993340, 1.53482580, 1.61109149, 2.01097894, 1.87727582, 1.34426224, 1.49639559, 1.93392670, 1.37553799],
         [1.49443316, 1.97318459, 2.19717979, 1.59254301, 1.68133974, 2.02672839, 2.01310015, 1.43589234, 1.69781375, 2.09603000, 1.44058943],
         [1.34929502, 1.71412206, 1.74131823, 1.41029775, 1.56077397, 1.71943927, 1.69139004, 1.23339808, 1.33667815, 1.73944271, 1.24329603]],

        [[1.75136340, 2.35909891, 1.66790223, 2.38119912, 0.64075530, 0.64075530, 0.64075530, 0.64075530, 0.64075530, 0.64075530, 0.64075530],
         [1.22181571, 1.78588963, 1.27680099, 1.59650970, 0.52781963, 0.52781963, 0.52781963, 0.52781963, 0.52781963, 0.52781963, 0.52781963],
         [1.64594960, 2.38568592, 1.83570778, 2.29080629, 0.64082849, 0.64082849, 0.64082849, 0.64082849, 0.64082849, 0.64082849, 0.64082849],
         [1.26192617, 1.64760089, 1.22102571, 1.73259580, 0.53460336, 0.53460336, 0.53460336, 0.53460336, 0.53460336, 0.53460336, 0.53460336],
         [0.50299925, 0.62577212, 0.51254773, 0.61388606, 0.33953226, 0.33953226, 0.33953226, 0.33953226, 0.33953226, 0.33953226, 0.33953226],
         [0.50299925, 0.62577212, 0.51254773, 0.61388606, 0.33953226, 0.33953226, 0.33953226, 0.33953226, 0.33953226, 0.33953226, 0.33953226],
         [0.50299925, 0.62577212, 0.51254773, 0.61388606, 0.33953226, 0.33953226, 0.33953226, 0.33953226, 0.33953226, 0.33953226, 0.33953226],
         [0.50299925, 0.62577212, 0.51254773, 0.61388606, 0.33953226, 0.33953226, 0.33953226, 0.33953226, 0.33953226, 0.33953226, 0.33953226],
         [0.50299925, 0.62577212, 0.51254773, 0.61388606, 0.33953226, 0.33953226, 0.33953226, 0.33953226, 0.33953226, 0.33953226, 0.33953226],
         [0.50299925, 0.62577212, 0.51254773, 0.61388606, 0.33953226, 0.33953226, 0.33953226, 0.33953226, 0.33953226, 0.33953226, 0.33953226],
         [0.50299925, 0.62577212, 0.51254773, 0.61388606, 0.33953226, 0.33953226, 0.33953226, 0.33953226, 0.33953226, 0.33953226, 0.33953226]]])
    bi_stochastic = paddle.to_tensor(s, dtype='float32', stop_gradient=False)
    n_rows = paddle.to_tensor([11,4])
    n_cols = paddle.to_tensor([11,4])

    model = Voting(2)
    
    with paddle.set_grad_enabled(True):
        m = model(bi_stochastic, n_rows, n_cols)
        loss = m.sum()
        
        paddle.autograd.backward([loss])
        print(bi_stochastic.grad)    

