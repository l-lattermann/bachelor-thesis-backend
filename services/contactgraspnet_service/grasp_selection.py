import numpy as np


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


def select_diverse_top_grasps(
    grasps,
    scores,
    contacts,
    openings,
    num_grasps,
    top_score_candidates,
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

    n_candidates = min(top_score_candidates, len(scores))
    order = np.argsort(scores)[::-1][:n_candidates]

    grasps_f = grasps[order]
    scores_f = scores[order]
    contacts_f = contacts[order]
    openings_f = openings[order] if len(openings) > 0 else np.array([], dtype=np.float32)

    selected_idx = [0]

    while len(selected_idx) < min(num_grasps, len(contacts_f)):
        selected_contacts = contacts_f[selected_idx]

        best_idx = None
        best_dist = -np.inf

        for i in range(len(contacts_f)):
            if i in selected_idx:
                continue

            dist = _pairwise_min_distance(contacts_f[i], selected_contacts)

            if dist > best_dist:
                best_dist = dist
                best_idx = i

        if best_idx is None:
            break

        selected_idx.append(best_idx)

    selected_idx = np.array(selected_idx[:num_grasps], dtype=np.int32)

    selected_grasps = grasps_f[selected_idx]
    selected_scores = scores_f[selected_idx]
    selected_contacts = contacts_f[selected_idx]
    selected_openings = openings_f[selected_idx] if len(openings_f) > 0 else np.array([], dtype=np.float32)

    return selected_grasps, selected_scores, selected_contacts, selected_openings


def process_contact_graspnet_result(result, sel_cfg):
    selected_grasps = {}
    selected_scores = {}
    selected_contacts = {}
    selected_openings = {}

    use_distance_div = bool(sel_cfg.get("distance_div_filtering", False))

    for key in result["pred_grasps_cam"]:
        grasps_k = _normalize_grasps(result["pred_grasps_cam"][key])
        scores_k = np.atleast_1d(result["scores"][key]).astype(np.float32)
        contacts_k = np.atleast_2d(result["contact_pts"][key]).astype(np.float32)
        openings_k = _normalize_openings(result["gripper_openings"][key], len(grasps_k))

        if use_distance_div:
            selected_g, selected_s, selected_c, selected_o = select_diverse_top_grasps(
                grasps=grasps_k,
                scores=scores_k,
                contacts=contacts_k,
                openings=openings_k,
                num_grasps=sel_cfg["num_grasps"],
                top_score_candidates=sel_cfg["top_score_candidates"],
            )
        else:
            selected_g, selected_s, selected_c, selected_o = select_top_grasps(
                grasps=grasps_k,
                scores=scores_k,
                contacts=contacts_k,
                openings=openings_k,
                num_grasps=sel_cfg["num_grasps"],
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