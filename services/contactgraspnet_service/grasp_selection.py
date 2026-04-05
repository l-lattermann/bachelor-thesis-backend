import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



def _pairwise_min_distance(point, selected_points):
    if len(selected_points) == 0:
        return np.inf
    dists = np.linalg.norm(selected_points - point[None, :], axis=1)
    return float(np.min(dists))


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


def k_means_clustering(
    grasps,
    scores,
    contacts,
    openings,
    num_grasps,
    kmeans_n_init=10,
    kmeans_random_state=0,
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

    k = min(num_grasps, len(contacts))
    if k == 0:
        return (
            np.empty((0, 4, 4), dtype=np.float32),
            np.array([], dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.array([], dtype=np.float32),
        )

    if k == 1:
        selected_idx = np.array([int(np.argmax(scores))], dtype=np.int32)
    else:
        approach = grasps[:, :3, 2]

        X = np.concatenate([
            contacts,
            orientation_weight * approach
        ], axis=1)

        X = StandardScaler().fit_transform(X)

        kmeans = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=kmeans_n_init,
            random_state=kmeans_random_state,
        )

        labels = kmeans.fit_predict(X)

        selected_idx = []
        for cluster_id in range(k):
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue
            best_local = cluster_indices[np.argmax(scores[cluster_indices])]
            selected_idx.append(best_local)

        selected_idx = np.array(selected_idx, dtype=np.int32)

        if len(selected_idx) > 0:
            selected_idx = selected_idx[np.argsort(scores[selected_idx])[::-1]]

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
    use_k_means=False,
    kmeans_n_init=10,
    kmeans_random_state=0,
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

        if use_k_means:
            selected_g, selected_s, selected_c, selected_o = k_means_clustering(
                grasps=grasps_k,
                scores=scores_k,
                contacts=contacts_k,
                openings=openings_k,
                num_grasps=num_grasps,
                kmeans_n_init=kmeans_n_init,
                kmeans_random_state=kmeans_random_state,
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