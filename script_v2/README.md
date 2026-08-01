# Token-efficient gaze-to-learning-element pipeline

The pipeline is intentionally split into two phases.

## Phase A — build the standard library once

Process all course videos together, detect visually unique states, and send only one representative image of each state to an AI model. A state is a slide/webpage plus its visible interaction layer; clicking a button and opening a popup creates a new state. The result is one `element_library.json` per course/version.

Each state contains:

- `slide_id`, `state_id`, and its video time interval;
- reusable element types such as webpage navigation, webpage panel/title, slide title, paragraph, image, button, popup, and slide navigation;
- a stable element instance ID and human-readable label;
- one or more normalized polygons. Multiple or concave polygons support irregular/disconnected shapes;
- `priority`, used when popup/button regions overlap base slide content.

`example_element_library.json` is the schema example. Rectangle-only coordinates are deliberately avoided.

Recommended state discovery before any AI call:

1. Crop the fixed course player region once.
2. Compute perceptual hashes or SSIM at scene-change candidates.
3. Cluster repeated states across every participant/video.
4. Call the AI only for one representative of each cluster, preferably as a batch.
5. Cache results by image hash. Re-running participant gaze must make zero AI calls.

The stable taxonomy is global, while coordinates and text are slide-state instances. This distinction matters: a `paragraph` type is reusable, but its polygon belongs to a particular slide/state.

## Phase B — map every participant locally

```bash
python script_v2/gaze_to_events.py \
  --library script_v2/example_element_library.json \
  --gaze-csv output/USER/gaze_coordinates.csv \
  --user-id USER \
  --output output/USER/gaze_events.csv
```

The output columns are:

`user_id, slide_id, state_id, time, learning_element, element_type, duration`

For the example “title from 6:00 until paragraph at 6:20”, the title row is equivalent to:

```text
USER,slide_01,slide_01_base,360.0,title,slide_title,20.0
```

Rows are run-length encoded from the full-frame gaze stream. Use `--min-duration 0.1` to discard very short detector jitter; keep it at `0` when raw transitions are required.

## Why the old refinement should not be in the production path

The old `scripts_image/step_3_refine_elements_gpt.py` assigns one AI label to an entire time bucket. That both spends tokens repeatedly and can erase real gaze transitions inside the bucket. The learned library should be AI-generated/verified once; gaze-to-polygon lookup and duration calculation should remain deterministic.
