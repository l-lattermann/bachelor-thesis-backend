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


def select_diverse_grasps(
    grasps,
    scores,
    contacts,
    openings,
    num_grasps,
    confidence_threshold=0.28,
    top_score_candidates=40,
    min_contact_distance=0.03,
    fallback_to_top_scores=True,
):
    grasps = _normalize_grasps(grasps)
    scores = np.atleast_1d(scores).astype(np.float32)
    contacts = np.atleast_2d(contacts).astype(np.float32)
    openings = _normalize_openings(openings, len(grasps))

    if len(scores) == 0 or len(contacts) == 0 or len(grasps) == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )

    # sort globally by score descending
    global_order = np.argsort(scores)[::-1]
    grasps_all = grasps[global_order]
    scores_all = scores[global_order]
    contacts_all = contacts[global_order]
    openings_all = openings[global_order] if len(openings) > 0 else np.array([])

    # 1) strict confidence filter
    keep = scores_all >= confidence_threshold
    grasps_f = grasps_all[keep]
    scores_f = scores_all[keep]
    contacts_f = contacts_all[keep]
    openings_f = openings_all[keep] if len(openings_all) > 0 else np.array([])

    # 2) controlled fallback if threshold removes everything
    if len(scores_f) == 0:
        if not fallback_to_top_scores:
            return (
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
            )

        fallback_n = min(top_score_candidates, len(scores_all))
        grasps_f = grasps_all[:fallback_n]
        scores_f = scores_all[:fallback_n]
        contacts_f = contacts_all[:fallback_n]
        openings_f = openings_all[:fallback_n] if len(openings_all) > 0 else np.array([])

    # 3) keep only strongest candidates before diversity selection
    order = np.argsort(scores_f)[::-1][:top_score_candidates]
    grasps_f = grasps_f[order]
    scores_f = scores_f[order]
    contacts_f = contacts_f[order]
    openings_f = openings_f[order] if len(openings_f) > 0 else np.array([])

    # 4) greedy diversity selection
    selected_idx = [0]

    while len(selected_idx) < min(num_grasps, len(contacts_f)):
        selected_contacts = contacts_f[selected_idx]

        best_candidate = None
        best_candidate_value = -np.inf

        for i in range(len(contacts_f)):
            if i in selected_idx:
                continue

            dist = _pairwise_min_distance(contacts_f[i], selected_contacts)

            if dist < min_contact_distance:
                continue

            value = dist + 1e-3 * float(scores_f[i])

            if value > best_candidate_value:
                best_candidate_value = value
                best_candidate = i

        if best_candidate is None:
            break

        selected_idx.append(best_candidate)

    # 5) fill with best remaining if needed
    if fallback_to_top_scores and len(selected_idx) < min(num_grasps, len(contacts_f)):
        for i in range(len(contacts_f)):
            if i not in selected_idx:
                selected_idx.append(i)
            if len(selected_idx) >= min(num_grasps, len(contacts_f)):
                break

    selected_idx = np.array(selected_idx[:num_grasps], dtype=np.int32)

    selected_grasps = grasps_f[selected_idx]
    selected_scores = scores_f[selected_idx]
    selected_contacts = contacts_f[selected_idx]
    selected_openings = openings_f[selected_idx] if len(openings_f) > 0 else np.array([])

    return selected_grasps, selected_scores, selected_contacts, selected_openings


def process_contact_graspnet_result(result, sel_cfg):
    """
    Processes the full result dict returned by cs.run_contact_graspnet(...).

    Applies selection per key and returns the same result dict structure
    with filtered values.
    """
    selected_grasps = {}
    selected_scores = {}
    selected_contacts = {}
    selected_openings = {}

    for key in result["pred_grasps_cam"]:
        grasps_k = _normalize_grasps(result["pred_grasps_cam"][key])
        scores_k = np.atleast_1d(result["scores"][key]).astype(np.float32)
        contacts_k = np.atleast_2d(result["contact_pts"][key]).astype(np.float32)
        openings_k = _normalize_openings(result["gripper_openings"][key], len(grasps_k))

        selected_g, selected_s, selected_c, selected_o = select_diverse_grasps(
            grasps=grasps_k,
            scores=scores_k,
            contacts=contacts_k,
            openings=openings_k,
            num_grasps=sel_cfg["num_grasps"],
            confidence_threshold=sel_cfg["confidence_threshold"],
            top_score_candidates=sel_cfg["top_score_candidates"],
            min_contact_distance=sel_cfg["min_contact_distance"],
            fallback_to_top_scores=sel_cfg.get("fallback_to_top_scores", True),
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