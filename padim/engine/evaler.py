# Copyright 2023 AlphaBetter Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Dict
from padim.utils.seed import init_seed
from omegaconf import OmegaConf, DictConfig


class Evaler:
    def __init__(self, config: DictConfig) -> None:
        self.config = config

    def validation(self) -> None:
        pass

gt_list = []
gt_mask_list = []
test_imgs = []

# extract test set features
for (x, y, mask) in tqdm(test_dataloader, f'| feature extraction | test | {class_name} |'):
    test_imgs.extend(x.cpu().detach().numpy())
    gt_list.extend(y.cpu().detach().numpy())
    gt_mask_list.extend(mask.cpu().detach().numpy())
    # model prediction
    with torch.no_grad():
        _ = model(x.to(device))
    # get intermediate layer outputs
    for k, v in zip(test_outputs.keys(), outputs):
        test_outputs[k].append(v.cpu().detach())
        # print(v.shape)
    # initialize hook outputs
    outputs = []
if args.npca or args.pca:
    for i, k in enumerate(test_outputs.keys()):
        outputs_reduced = []
        for batch in test_outputs[k]:
            reduced = pca_reduction(batch, pca_mean[i], pca_components[i], device)
            outputs_reduced.append(reduced)
        test_outputs[k] = torch.cat(outputs_reduced, 0).cpu().detach()
else:
    for k, v in test_outputs.items():
        test_outputs[k] = torch.cat(v, 0)

# Embedding concat
embedding_vectors = test_outputs['layer1']
for layer_name in ['layer2', 'layer3']:
    embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])

# randomly select d dimension
if args.reduce_dim:
    embedding_vectors = torch.index_select(embedding_vectors, 1, idx)

# calculate distance matrix
B, C, H, W = embedding_vectors.size()
if use_gpu:
    embedding_vectors = embedding_vectors.view(B, C, H * W).to(device)
    dist_list = torch.zeros(size=(H * W, B))
    mean = torch.Tensor(train_outputs[0]).to(device)
    cov_inv = torch.Tensor(train_outputs[1]).to(device)
    if args.save_gpu_memory:
        for i in range(H * W):
            delta = embedding_vectors[:, :, i] - mean[:, i]
            m_dist = torch.sqrt(torch.diag(torch.mm(torch.mm(delta, cov_inv[:, :, i]), delta.t())).clamp(0))
            dist_list[i] = m_dist
        dist_list = dist_list.cpu().numpy()
        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)
        dist_list = torch.tensor(dist_list)
    else:
        delta = (embedding_vectors - mean).permute(2, 0, 1)
        dist_list = (torch.matmul(delta, cov_inv.permute(2, 0, 1)) * delta).sum(2).permute(1, 0)
        dist_list = dist_list.reshape(B, H, W)
        dist_list = dist_list.clamp(0).sqrt().cpu()
else:
    embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
    dist_list = []
    for i in range(H * W):
        mean = train_outputs[0][:, i]
        # dist = [mahalanobis(sample[:, i], mean, train_outputs[1][:, :, i]) for sample in embedding_vectors]
        dist = SSD.cdist(embedding_vectors[:, :, i], mean[None, :], metric='mahalanobis', VI=train_outputs[1][:, :, i])
        dist = list(itertools.chain(*dist))
        dist_list.append(dist)
    dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)
    dist_list = torch.tensor(dist_list)
# upsample
score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',
                          align_corners=False).squeeze().numpy()
# apply gaussian smoothing on the score map
for i in range(score_map.shape[0]):
    score_map[i] = gaussian_filter(score_map[i], sigma=4)
# Normalization
max_score = score_map.max()
min_score = score_map.min()
scores = (score_map - min_score) / (max_score - min_score)
# calculate image-level ROC AUC score
img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
gt_list = np.asarray(gt_list)
fpr, tpr, _ = roc_curve(gt_list, img_scores)
img_roc_auc = roc_auc_score(gt_list, img_scores)
total_roc_auc.append(img_roc_auc)
print('image ROCAUC: %.3f' % (img_roc_auc))
fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc))

# get optimal threshold
gt_mask = np.asarray(gt_mask_list)
precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
a = 2 * precision * recall
b = precision + recall
f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
threshold = thresholds[np.argmax(f1)]

# calculate per-pixel level ROCAUC
fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
total_pixel_roc_auc.append(per_pixel_rocauc)
print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
save_dir = args.save_path + '/' + f'pictures_{args.arch}'
os.makedirs(save_dir, exist_ok=True)
plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, class_name)

print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
fig_img_rocauc.legend(loc="lower right")

print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))
fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
fig_pixel_rocauc.legend(loc="lower right")

fig.tight_layout()
fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)