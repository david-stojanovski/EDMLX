import numpy as np
import torch
import torchvision
from skimage.metrics import hausdorff_distance
from sklearn.metrics import classification_report, roc_auc_score
from torchmetrics.functional import dice
from tqdm import tqdm

from torch_utils import misc


def calc_la_dice(prediction, target):
    return dice(prediction == 3, target == 3, num_classes=2, average='macro', ignore_index=0)


def calc_lv_endo_dice(prediction, target):
    return dice(prediction == 1, target == 1, num_classes=2, average='macro', ignore_index=0)


def calc_lv_epi_dice(prediction, target):
    pred_bool_mask = prediction == 1
    pred_bool_mask += prediction == 2
    target_bool_mask = target == 1
    target_bool_mask += target == 2
    return dice(pred_bool_mask, target_bool_mask, num_classes=2, average='macro', ignore_index=0)


def calc_lv_endo_hausdorff_distance(prediction, target):
    lv_endo_pred = prediction == 1
    lv_endo_gt = target == 1
    return hausdorff_distance(lv_endo_pred.float().cpu().numpy(), lv_endo_gt.float().cpu().numpy())


def train_segmentation_cycle(model, train_loader, data_transforms, num_classes, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    with torch.amp.autocast('cuda'), tqdm(train_loader, unit="batch") as tepoch:
        for imgs, segmentations in tepoch:
            imgs, segmentations = imgs.to(device), segmentations.to(device)
            if data_transforms is not None:
                segmentations = torchvision.tv_tensors.Mask(segmentations)
                imgs, segmentations = data_transforms(imgs, segmentations)
                segmentations = segmentations.float()
            imgs = imgs / 255.

            optimizer.zero_grad(set_to_none=True)
            segmentation_preds = model(imgs)
            batch_loss = criterion(segmentation_preds,
                                   torch.nn.functional.one_hot(segmentations.long(),
                                                               num_classes).permute(0, 3, 1, 2).float())

            segmentation_preds = torch.argmax(segmentation_preds, dim=1)
            dice_la = calc_la_dice(segmentation_preds, segmentations).cpu().numpy()
            dice_lv_endo = calc_lv_endo_dice(segmentation_preds, segmentations).cpu().numpy()
            dice_lv_epi = calc_lv_epi_dice(segmentation_preds, segmentations).cpu().numpy()

            mean_batch_dice = np.mean([dice_la, dice_lv_endo, dice_lv_epi])

            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += batch_loss.item()
            running_dice += mean_batch_dice

            tepoch.set_postfix(loss=batch_loss.item(), mean_dice=mean_batch_dice)

    return running_loss / len(train_loader), running_dice / len(train_loader)


def evaluate_segmentation_cycle(data_loader, data_transforms, device, model):
    model.eval()
    individual_dice_scores = []
    individual_hd_scores = []
    with torch.amp.autocast('cuda'), tqdm(data_loader, unit="batch") as tepoch:
        for imgs, segmentations in tepoch:
            imgs, segmentations = imgs.to(device), segmentations.to(device)
            if data_transforms is not None:
                segmentations = torchvision.tv_tensors.Mask(segmentations)
                imgs, segmentations = data_transforms(imgs, segmentations)
            imgs = imgs / 255.

            segmentation_preds = model(imgs)

            segmentation_preds = torch.argmax(segmentation_preds, dim=1)

            for sample in range(imgs.size(0)):
                dice_la = calc_la_dice(segmentation_preds[sample], segmentations[sample].float()).cpu().numpy()
                dice_lv_endo = calc_lv_endo_dice(segmentation_preds[sample],
                                                 segmentations[sample].float()).cpu().numpy()
                individual_hd_scores += [
                    calc_lv_endo_hausdorff_distance(segmentation_preds[sample], segmentations[sample])]
                dice_lv_epi = calc_lv_epi_dice(segmentation_preds[sample], segmentations[sample].float()).cpu().numpy()

                individual_dice_scores += [[dice_la, dice_lv_endo, dice_lv_epi]]

    individual_dice_scores = np.squeeze(individual_dice_scores)
    mean_per_label_dice = np.mean(individual_dice_scores, axis=0)
    std_per_label_dice = np.std(individual_dice_scores, axis=0)
    overall_dice_mean = np.mean(mean_per_label_dice)
    mean_hd = np.mean(individual_hd_scores)
    std_hd = np.std(individual_hd_scores)
    return overall_dice_mean, mean_per_label_dice, std_per_label_dice, mean_hd, std_hd


def train_classifier_cycle(model, train_loader, data_transforms, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0

    all_preds = []
    all_targets = []
    with torch.amp.autocast('cuda'), tqdm(train_loader, unit="batch") as tepoch:
        for imgs, class_labels in tepoch:
            imgs, class_labels = imgs.to(device), class_labels.to(device)
            if data_transforms is not None:
                imgs = data_transforms(imgs)
            imgs = imgs / 255.

            optimizer.zero_grad(set_to_none=True)
            class_preds = model(imgs)
            batch_loss = criterion(torch.squeeze(class_preds), class_labels)

            all_preds += [class_preds]
            all_targets += [class_labels]

            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += batch_loss.item() * imgs.size(0)

            tepoch.set_postfix(loss=batch_loss.item())

    all_preds = torch.argmax(torch.cat(all_preds), dim=1).cpu().numpy()
    report = classification_report(torch.cat(all_targets).cpu().numpy(),
                                   all_preds,
                                   target_names=['2ch', '4ch'],
                                   output_dict=True,
                                   zero_division=0.0)
    return running_loss / len(train_loader.dataset), report


def evaluate_classifier_cycle(model, test_loader, data_transforms, num_classes, device):
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for imgs, class_labels in test_loader:
            imgs, class_labels = imgs.to(device), class_labels.to(device)
            if data_transforms is not None:
                imgs = data_transforms(imgs)
            imgs = imgs / 255.

            all_preds += [model(imgs)]
            all_targets += [class_labels]

    all_preds = torch.argmax(torch.cat(all_preds), dim=1).cpu().numpy()
    report = classification_report(torch.cat(all_targets).cpu().numpy(), all_preds, target_names=['2ch', '4ch'],
                                   output_dict=True, zero_division=0.0)
    auroc_score = roc_auc_score(torch.cat(all_targets).cpu().numpy(), all_preds)
    return report, auroc_score


def train_autoencoder_cycle(args, epoch, model, perceptual_loss, data_loader, data_transforms, criterion, optimizer_g,
                            lr_scheduler_g, scaler_g, rank, tb_writer):
    model.train()

    total_epoch_loss = 0
    epoch_recon_loss = 0
    epoch_recon_loss_latent = 0

    epoch_kl_loss = 0

    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), disable=not rank == 0)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, data in progress_bar:
        images, masks = data
        images = images.to(rank, dtype=torch.bfloat16)
        masks = masks.to(rank).type(torch.bool) * 1.

        optimizer_g.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=args.amp, dtype=torch.bfloat16):
            masks = torchvision.tv_tensors.Mask(masks)
            images = images / 255.
            images, masks = data_transforms(images, masks)
            masks = masks.unsqueeze(1)

            recon, z_latent, kl_loss = model.module(images, masks)
            kl_loss = args.kl_weight * kl_loss

            if step == len(progress_bar) - 1 and rank == 0:
                tb_writer.add_image('input_image', images[0], epoch)
                tb_writer.add_image('recon_image', recon.clamp(0, 1)[0], epoch)
                tb_writer.add_image('latent_space', z_latent[0], epoch)

            if epoch == 0 and step == 0 and rank == 0:
                print(f"latent shape: {z_latent.shape}")
                print(f'Latent space data range: {z_latent.min()} to {z_latent.max()}')
                print(f"reconstruction shape: {recon.shape}")

            assert recon.max() > 0.1

            recons_loss = criterion(recon, images) * args.mse_weight

            if epoch >= 5:
                p_loss = perceptual_loss(recon.float(), images.float()).clamp(1e-7, 1e8) * args.perceptual_weight
                recons_loss_latent = criterion(z_latent,
                                               torch.nn.functional.interpolate(images, z_latent.shape[
                                                   -1])) * args.mse_latent_weight
            else:
                p_loss = 0
                recons_loss_latent = 0

            del recon, z_latent

            loss_g = recons_loss + p_loss + kl_loss + recons_loss_latent

        scaler_g.scale(loss_g).backward()
        scaler_g.unscale_(optimizer_g)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        scaler_g.step(optimizer_g)
        scaler_g.update()
        lr_scheduler_g.step()

        total_epoch_loss += loss_g.item()
        epoch_kl_loss += kl_loss.item()
        epoch_recon_loss += recons_loss.item()
        epoch_recon_loss_latent += recons_loss_latent

        assert epoch_kl_loss > 0
        assert total_epoch_loss > 0

        progress_bar.set_postfix(
            {
                'total_loss': total_epoch_loss / (step + 1),
                'recon_loss': (epoch_recon_loss / (step + 1)),
                'kl_loss': (epoch_kl_loss / (step + 1)),
                'p_loss': (p_loss / (step + 1)),
                'recon_loss_latent': (epoch_recon_loss_latent / (step + 1)),
            }
        )
    progress_bar.close()
    return total_epoch_loss, epoch_recon_loss, epoch_kl_loss, epoch_recon_loss_latent


def evaluate_autoencoder_cycle(args, epoch, model, perceptual_loss, data_loader, data_transforms, criterion, rank,
                               tb_writer):
    model.eval()
    total_epoch_loss = 0
    val_recon_epoch_loss = 0
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=args.amp, dtype=torch.bfloat16):
        for step, data in enumerate(data_loader):
            images, masks = data
            images = images.to(rank, dtype=torch.bfloat16) / 255.
            masks = torchvision.tv_tensors.Mask(masks)
            images, masks = data_transforms(images, masks)
            masks = masks.to(rank).type(torch.bool).unsqueeze(1) * 1.

            recon, z_latent, kl_loss = model(images, masks)
            kl_loss = args.kl_weight * kl_loss

            recons_loss = criterion(recon, images) * args.mse_weight
            val_recon_epoch_loss += recons_loss.item()

            if step == 0 and rank == 0:
                tb_writer.add_image('input_image', images[0], epoch)
                tb_writer.add_image('recon_image', recon.clamp(0, 1)[0], epoch)
                tb_writer.add_image('latent_space', z_latent[0], epoch)

            if args.perceptual_weight > 0 and epoch >= 5:
                p_loss = perceptual_loss(recon.float(),
                                         images.float()).clamp(1e-7, 1e8) * args.perceptual_weight
            else:
                p_loss = 0

            total_epoch_loss += recons_loss + p_loss + kl_loss

        return total_epoch_loss, recons_loss, kl_loss


def train_diffusion_cycle(ddp, net, dataset_iterator, device, loss_fn, optimizer, ema, training_stats, loss_scaling,
                          num_accumulation_rounds, batch_gpu_total, vae_kwargs, augment_pipe, network_kwargs,
                          dataset_kwargs, autoencoder_ddp, ema_halflife_kimg, ema_rampup_ratio, cur_nimg, batch_size):
    net.train().requires_grad_(True)
    optimizer.zero_grad(set_to_none=True)
    for round_idx in range(num_accumulation_rounds):
        with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)), torch.amp.autocast('cuda'):
            if dataset_kwargs.use_semantic_labels:
                images, labels, labelmaps, fname = next(dataset_iterator)
                labelmaps = torch.squeeze(labelmaps.to(device), dim=1)
            else:
                images, labels, fname = next(dataset_iterator)
                labelmaps = None
            images = images.to(device, dtype=torch.float16) / 255.  # gets scaled to [-1, 1] in augmentations
            labels = labels.to(device)
            if vae_kwargs:
                with torch.no_grad():
                    assert labelmaps is not None
                    masks_bool = torch.argmax(labelmaps, dim=1).unsqueeze(1).to(torch.bool) * 1.
                    images = autoencoder_ddp.encode_stage_2(images, masks_bool)
                    labelmaps = torch.nn.functional.interpolate(labelmaps,
                                                                size=images.shape[-2:],
                                                                mode='nearest')

            assert images.shape[-1] == dataset_kwargs['diffusion_resolution'], 'Image resolution mismatch'

            loss = loss_fn(net=ddp,
                           images=images,
                           labels=labels,
                           semantic_labels=labelmaps,
                           semantic_cond=network_kwargs.semantic_cond,
                           augment_pipe=augment_pipe)
            training_stats.report('Loss/loss', loss)
            loss.sum().mul(loss_scaling / batch_gpu_total).backward()

    # Update weights.

    for param in net.parameters():
        if param.grad is not None:
            torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
    optimizer.step()

    # Update EMA.
    ema_halflife_nimg = ema_halflife_kimg * 1000
    if ema_rampup_ratio is not None:
        ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
    ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
    for p_ema, p_net in zip(ema.parameters(), net.parameters()):
        p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
    return


def validate_diffusion_cycle(
        ddp_net=None,
        autoencoder_ddp=None,
        dataset_iterator_val=None,
        loss_fn=None,  # Options for loss function.
        network_kwargs=None,  # Options for model and preconditioning.
        dataset_kwargs=None,  # Options for dataset and augmentations.
        total_kimg_val=None,
        aug_pipe_val=None,  # Options for validation augmentation pipeline, None = disable.
        num_accumulation_rounds=None,
        batch_size=None,
        input_training_stats=None,
        device=None,
):
    ddp_net.eval().requires_grad_(False).to(device)

    cur_tick = 0
    cur_nimg = 0

    while True:
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for round_idx in range(num_accumulation_rounds):
                with misc.ddp_sync(ddp_net, (round_idx == num_accumulation_rounds - 1)):
                    if dataset_kwargs.use_semantic_labels:
                        images, labels, labelmaps, fname = next(dataset_iterator_val)
                        labelmaps = torch.squeeze(labelmaps.to(device), dim=1)
                    else:
                        images, labels, fname = next(dataset_iterator_val)
                        labelmaps = None
                    images = images.to(device, dtype=torch.float16) / 255.
                    labels = labels.to(device)
                    if autoencoder_ddp is not None:
                        with torch.no_grad():
                            masks_bool = torch.argmax(labelmaps, dim=1).unsqueeze(1).to(torch.bool) * 1.
                            images = autoencoder_ddp.encode_stage_2(images, masks_bool)

                            labelmaps = torch.nn.functional.interpolate(labelmaps,
                                                                        size=images.shape[-2:],
                                                                        mode='nearest')

                    val_loss = loss_fn(net=ddp_net, images=images, labels=labels, semantic_labels=labelmaps,
                                       semantic_cond=network_kwargs.semantic_cond, augment_pipe=aug_pipe_val)
                    input_training_stats.report('Val_loss/val_loss', val_loss)

            cur_nimg += batch_size

            done = (cur_nimg >= total_kimg_val * 1000)
            cur_tick += 1
            if done:
                return input_training_stats
