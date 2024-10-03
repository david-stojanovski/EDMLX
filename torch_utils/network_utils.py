import copy
import json
import os
import pickle
import time

import psutil
import torch

import dnnlib
from torch_utils import distributed as dist
from torch_utils import misc


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_kernels_strides(img_size):
    """
    This function is only used for decathlon datasets with the provided patch sizes.
    When refering this method for other tasks, please ensure that the patch size for each spatial dimension should
    be divisible by the product of all strides in the corresponding dimension.
    In addition, the minimal spatial size should have at least one dimension that has twice the size of
    the product of all strides. For patch sizes that cannot find suitable strides, an error will be raised.

    """
    sizes, spacings = [img_size, img_size], [1.0, 1.0]
    input_size = sizes
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
                )
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides


def pretty_classification_report(data_dict):
    # taken from: https://gist.github.com/geblanco/5cfe4a3224e021113968a8c7ebe31419
    """Build a text report showing the main classification metrics.
    Read more in the :ref:`User Guide <classification_report>`.
    Parameters
    ----------
    report : string
        Text summary of the precision, recall, F1 score for each class.
        Dictionary returned if output_dict is True. Dictionary has the
        following structure::
            {'label 1': {'precision':0.5,
                         'recall':1.0,
                         'f1-score':0.67,
                         'support':1},
             'label 2': { ... },
              ...
            }
        The reported averages include macro average (averaging the unweighted
        mean per label), weighted average (averaging the support-weighted mean
        per label), and sample average (only for multilabel classification).
        Micro average (averaging the total true positives, false negatives and
        false positives) is only shown for multi-label or multi-class
        with a subset of classes, because it corresponds to accuracy otherwise.
        See also :func:`precision_recall_fscore_support` for more details
        on averages.
        Note that in binary classification, recall of the positive class
        is also known as "sensitivity"; recall of the negative class is
        "specificity".
    """

    non_label_keys = ["accuracy", "macro avg", "weighted avg"]
    y_type = "binary"
    digits = 2

    target_names = [
        "%s" % key for key in data_dict.keys() if key not in non_label_keys
    ]

    # labelled micro average
    micro_is_accuracy = (y_type == "multiclass" or y_type == "binary")

    headers = ["precision", "recall", "f1-score", "support"]
    p = [data_dict[l][headers[0]] for l in target_names]
    r = [data_dict[l][headers[1]] for l in target_names]
    f1 = [data_dict[l][headers[2]] for l in target_names]
    s = [data_dict[l][headers[3]] for l in target_names]

    rows = zip(target_names, p, r, f1, s)

    if y_type.startswith("multilabel"):
        average_options = ("micro", "macro", "weighted", "samples")
    else:
        average_options = ("micro", "macro", "weighted")

    longest_last_line_heading = "weighted avg"
    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, len(longest_last_line_heading), digits)
    head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
    report = head_fmt.format("", *headers, width=width)
    report += "\n\n"
    row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"
    for row in rows:
        report += row_fmt.format(*row, width=width, digits=digits)
    report += "\n"

    # compute all applicable averages
    for average in average_options:
        if average.startswith("micro") and micro_is_accuracy:
            line_heading = "accuracy"
        else:
            line_heading = average + " avg"

        if line_heading == "accuracy":
            avg = [data_dict[line_heading], sum(s)]
            row_fmt_accuracy = "{:>{width}s} " + \
                               " {:>9.{digits}}" * 2 + " {:>9.{digits}f}" + \
                               " {:>9}\n"
            report += row_fmt_accuracy.format(line_heading, "", "",
                                              *avg, width=width,
                                              digits=digits)
        else:
            avg = list(data_dict[line_heading].values())
            report += row_fmt.format(line_heading, *avg,
                                     width=width, digits=digits)
    return report


def save_model_config(_args, _save_dir):
    with open(os.path.join(_save_dir, 'model_config.json'), 'w') as f:
        argparse_dict = vars(_args)
        json.dump(argparse_dict, f, indent=4)


def append_diffusion_train_output(stats, cur_tick, cur_nimg, tick_start_nimg, total_kimg, start_time, tick_start_time,
                                  maintenance_time, device):
    tick_end_time = time.time()
    if cur_tick != 0:
        eta = (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * (total_kimg * 1000 - cur_nimg)
    else:
        eta = 0
    fields = []
    fields += [f"tick {stats.report0('Progress/tick', cur_tick):<5d}"]
    fields += [f"kimg {stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
    fields += [
        f"time {dnnlib.util.format_time(stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
    fields += [f"sec/tick {stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
    fields += [
        f"sec/kimg {stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
    fields += [f"ETA {dnnlib.util.format_time(eta):<12s}"]
    fields += [f"maintenance {stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
    fields += [
        f"cpumem {stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2 ** 30):<6.2f}"]
    fields += [
        f"gpumem {stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2 ** 30):<6.2f}"]
    fields += [
        f"reserved {stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2 ** 30):<6.2f}"]
    return fields, tick_end_time


def diffusion_load_from_pkl(resume_pkl, net, ema):
    dist.print0(f'Loading network weights from "{resume_pkl}"...')
    if dist.get_rank() != 0:
        torch.distributed.barrier()  # rank 0 goes first
    with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
        data = pickle.load(f)
    if dist.get_rank() == 0:
        torch.distributed.barrier()  # other ranks follow
    misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
    misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
    return


def diffusion_load_from_state_dump(resume_state_dump, net, optimizer):
    dist.print0(f'Loading training state from "{resume_state_dump}"...')
    data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
    misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
    optimizer.load_state_dict(data['optimizer_state'])
    return


def pkl_dump_best_iter(ema, loss_fn, augment_pipe, dataset_kwargs, savepath):
    data = dict(ema=ema,
                loss_fn=loss_fn,
                augment_pipe=augment_pipe,
                dataset_kwargs=dict(dataset_kwargs))
    for key, value in data.items():
        if isinstance(value, torch.nn.Module):
            value = copy.deepcopy(value).eval().requires_grad_(False)
            misc.check_ddp_consistency(value)
            data[key] = value.cpu()
        del value  # conserve memory
    if dist.get_rank() == 0:
        with open(savepath, 'wb') as f:
            pickle.dump(data, f)
    return


def prepare_data4net(semantic_labels, augment_pipe, images):
    if semantic_labels is not None:
        if augment_pipe is not None:
            y, augment_labels, augment_segmaps = augment_pipe(images, semantic_labels)
        else:
            y, augment_labels, augment_segmaps = images, None, semantic_labels
    else:
        if augment_pipe is not None:
            y, augment_labels, augment_segmaps = augment_pipe(images)
        else:
            y, augment_labels, augment_segmaps = images, None, None

    return y, augment_labels, augment_segmaps
