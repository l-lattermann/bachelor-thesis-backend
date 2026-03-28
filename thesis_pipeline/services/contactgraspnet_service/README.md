curl -X POST http://localhost:8002/inference \
  -H "Content-Type: application/json" \
  -d '{
    "npz_path": "/shared/pipeline_io/uois_output.npz"
  }' \
  --output cgn_result.npz











## Explanation of Contact-GraspNet Output and Filtering Strategy

The output of the predict_scene_grasps(...) function represents a set of grasp candidates generated for a given scene based on the input point cloud. The results are structured as dictionaries, where each key corresponds to a segmented object or region in the scene. In this example, only a single key (key = 3) is present, indicating that grasps were generated for one segmented object.

For this object, a total of 85 grasp candidates were predicted. Each grasp is represented as a homogeneous transformation matrix of shape (4, 4), resulting in an overall tensor of shape (85, 4, 4). These matrices encode the full 6-DoF pose of the gripper in the camera coordinate system, where the upper-left 3×3 submatrix represents rotation and the last column represents translation.

In addition to the grasp poses, three associated outputs are provided:
	•	Scores ((85,)): Each grasp is assigned a scalar confidence value indicating its predicted quality or success likelihood. Higher values correspond to more promising grasps.
	•	Contact Points ((85, 3)): These represent the 3D positions of the predicted grasp contact locations in the camera frame.
	•	Gripper Openings ((85,)): These values indicate the required gripper width for executing each grasp.

All four outputs are aligned such that each index corresponds to the same grasp candidate. This is confirmed by the consistency check, where all arrays have length 85.



## Need for Filtering

The raw output of Contact-GraspNet typically contains a large number of grasp candidates per object (in this case, 85). While this dense sampling ensures high coverage of possible grasp configurations, it is not suitable for downstream execution or visualization. Therefore, a filtering and selection step is required to reduce the set to a manageable number of high-quality grasps.

⸻
## Filtering Strategy

The filtering process can be divided into two stages:

1. Confidence-Based Filtering
A first reduction can be achieved by applying a threshold on the predicted scores. For example:
	•	Remove grasps with scores below a threshold (e.g., 0.25)
	•	This eliminates low-confidence and potentially unstable grasps

Alternatively, grasps can be ranked by score and only the top-k candidates retained.

2. Spatial Diversity Filtering (Farthest Point Sampling)
Since many high-scoring grasps may be spatially redundant (i.e., clustered around the same region), a second filtering step is applied to enforce diversity. This is typically done using a farthest point sampling strategy, which selects grasps that are maximally spread in space based on their contact points.

This ensures that:
	•	Selected grasps are not overlapping
	•	Different grasp approaches are considered
	•	The final set covers the object more uniformly

⸻

## Final Selection

After filtering, a fixed number of grasps (e.g., num_grasps = 5) is selected. This selection is typically performed per object or segment. In this example, although 85 grasps were initially generated, only a small subset is retained for further processing.

⸻

## Summary

The Contact-GraspNet output provides a dense set of grasp hypotheses, each defined by pose, confidence, contact location, and gripper width. Due to the large number of candidates, a two-stage filtering approach—based on confidence and spatial diversity—is required to obtain a compact and meaningful set of grasps suitable for execution or further reasoning.