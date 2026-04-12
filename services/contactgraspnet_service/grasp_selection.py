from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler



def _normalize_grasps(grasps):
    grasps = np.asarray(grasps)
    if grasps.size == 0:
        return np.empty((0, 4, 4), dtype=np.float32)
    if grasps.ndim == 2 and grasps.shape == (4, 4):
        return grasps[None, ...]
    return grasps


def _normalize_openings(openings, n_expected):
    openings = np.asarray(openings)
    if openings.ndim == 0 or openings.size == 0:
        return np.array([], dtype=np.float32)
    openings = np.atleast_1d(openings).astype(np.float32)
    if len(openings) != n_expected:
        return np.array([], dtype=np.float32)
    return openings


def select_top_grasps(
    grasps,
    scores,
    contacts,
    openings,
    num_grasps,
):
    grasps = _normalize_grasps(grasps)
    scores = np.atleast_1d(scores).astype(np.float32)
    contacts = np.atleast_2d(contacts).astype(np.float32)
    openings = _normalize_openings(openings, len(grasps))

    if len(grasps) == 0 or len(scores) == 0 or len(contacts) == 0:
        return (
            np.empty((0, 4, 4), dtype=np.float32),
            np.array([], dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.array([], dtype=np.float32),
        )

    n = min(num_grasps, len(scores))
    order = np.argsort(scores)[::-1][:n]

    selected_grasps = grasps[order]
    selected_scores = scores[order]
    selected_contacts = contacts[order]
    selected_openings = openings[order] if len(openings) > 0 else np.array([], dtype=np.float32)

    return selected_grasps, selected_scores, selected_contacts, selected_openings


def dbscan_clustering(
    grasps,
    scores,
    contacts,
    openings,
    num_grasps,
    min_score=0.0,
    eps=0.8,
    min_samples=1,
    orientation_weight=0.3,
):
    grasps = _normalize_grasps(grasps)
    scores = np.atleast_1d(scores).astype(np.float32)
    contacts = np.atleast_2d(contacts).astype(np.float32)
    openings = _normalize_openings(openings, len(grasps))

    if len(grasps) == 0 or len(scores) == 0 or len(contacts) == 0:
        return (
            np.empty((0, 4, 4), dtype=np.float32),
            np.array([], dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.array([], dtype=np.float32),
        )

    original_best_idx = int(np.argmax(scores))

    keep = scores >= float(min_score)

    grasps = grasps[keep]
    scores = scores[keep]
    contacts = contacts[keep]
    openings = openings[keep] if len(openings) > 0 else openings

    if len(grasps) == 0:
        return (
            np.empty((0, 4, 4), dtype=np.float32),
            np.array([], dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.array([], dtype=np.float32),
        )

    best_idx = int(np.argmax(scores))

    approach = grasps[:, :3, 2]

    X = np.concatenate(
        [
            contacts,
            orientation_weight * approach,
        ],
        axis=1,
    )

    X = StandardScaler().fit_transform(X)

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)

    selected_idx = []

    unique_labels = [lab for lab in np.unique(labels) if lab != -1]

    for cluster_id in unique_labels:
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue

        best_local = cluster_indices[np.argmax(scores[cluster_indices])]
        selected_idx.append(best_local)

    if len(selected_idx) == 0:
        print("CLUSTERING FAILED")
        selected_idx = np.argsort(scores)[::-1][:num_grasps]
        selected_idx = np.array(selected_idx, dtype=np.int32)

    else:
        selected_idx = np.array(selected_idx, dtype=np.int32)
        selected_idx = selected_idx[np.argsort(scores[selected_idx])[::-1]]

        if len(selected_idx) > num_grasps:
            candidate_contacts = contacts[selected_idx]

            chosen = [0]
            remaining = list(range(1, len(selected_idx)))

            while len(chosen) < num_grasps and remaining:
                chosen_pts = candidate_contacts[chosen]
                remaining_pts = candidate_contacts[remaining]

                dists = np.linalg.norm(
                    remaining_pts[:, None, :] - chosen_pts[None, :, :],
                    axis=2,
                )

                min_dists = dists.min(axis=1)
                best_remaining_pos = int(np.argmax(min_dists))

                chosen.append(remaining[best_remaining_pos])
                remaining.pop(best_remaining_pos)

            selected_idx = selected_idx[np.array(chosen, dtype=np.int32)]

    if best_idx not in selected_idx:
        if len(selected_idx) < num_grasps:
            selected_idx = np.append(selected_idx, best_idx)
        else:
            worst_pos = int(np.argmin(scores[selected_idx]))
            selected_idx[worst_pos] = best_idx

    selected_idx = np.unique(selected_idx)
    selected_idx = selected_idx[np.argsort(scores[selected_idx])[::-1]]

    if len(selected_idx) > num_grasps:
        selected_idx = selected_idx[:num_grasps]

    selected_grasps = grasps[selected_idx]
    selected_scores = scores[selected_idx]
    selected_contacts = contacts[selected_idx]
    selected_openings = openings[selected_idx] if len(openings) > 0 else np.array([], dtype=np.float32)

    return selected_grasps, selected_scores, selected_contacts, selected_openings


def process_contact_graspnet_result(
    result,
    num_grasps,
    top_score_candidates,
    use_dbscan=False,
    dbscan_min_score=0.0,
    dbscan_eps=0.8,
    dbscan_min_samples=1,
    orientation_weight=1.0,
):
    selected_grasps = {}
    selected_scores = {}
    selected_contacts = {}
    selected_openings = {}

    for key in result["pred_grasps_cam"]:
        grasps_k = _normalize_grasps(result["pred_grasps_cam"][key])
        scores_k = np.atleast_1d(result["scores"][key]).astype(np.float32)
        contacts_k = np.atleast_2d(result["contact_pts"][key]).astype(np.float32)
        openings_k = _normalize_openings(result["gripper_openings"][key], len(grasps_k))

        if use_dbscan:
            selected_g, selected_s, selected_c, selected_o = dbscan_clustering(
                grasps=grasps_k,
                scores=scores_k,
                contacts=contacts_k,
                openings=openings_k,
                num_grasps=num_grasps,
                min_score=dbscan_min_score,
                eps=dbscan_eps,
                min_samples=dbscan_min_samples,
                orientation_weight=orientation_weight,
            )
        else:
            selected_g, selected_s, selected_c, selected_o = select_top_grasps(
                grasps=grasps_k,
                scores=scores_k,
                contacts=contacts_k,
                openings=openings_k,
                num_grasps=num_grasps,
            )

        selected_grasps[key] = selected_g
        selected_scores[key] = selected_s
        selected_contacts[key] = selected_c
        selected_openings[key] = selected_o

    result["pred_grasps_cam"] = selected_grasps
    result["scores"] = selected_scores
    result["contact_pts"] = selected_contacts
    result["gripper_openings"] = selected_openings

    return result


def save_selected_cgn_output(
    output_path,
    pred_grasps_cam,
    scores,
    gripper_openings,
    pc_full,
    segmap,
    rgb,
    pc_colors,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        pred_grasps_cam=np.array(pred_grasps_cam, dtype=object),
        scores=np.array(scores, dtype=object),
        gripper_openings=gripper_openings,
        pc_full=pc_full,
        segmap=segmap,
        rgb=rgb,
        pc_colors=pc_colors,
    )

    return str(output_path)