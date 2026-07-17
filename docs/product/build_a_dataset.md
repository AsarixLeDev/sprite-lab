# Build a dataset

A dataset is the collection of images Sprite Lab prepares for learning. Your original files stay where they are.

## 1. Choose an image folder

Choose a folder containing PNG, JPEG, or WebP images. You do not need to type an internal file location or technical identifier.

If Sprite Lab already imported the folder, choose **Use existing imported dataset**. The native picker validates the
existing dataset and opens its review and labeling workflow without rebuilding or modifying it. The selection is
remembered for the current project.

Selection and manifest validation run as a background job. The Dataset page remains responsive and shows timestamped
progress from folder selection through validation and project activation. When validation finishes, the primary action
changes to **Use selected dataset**. It activates the existing dataset without rebuilding it, then exposes the review
and labeling actions.

If Sprite Lab cannot read an image, it leaves that image out and shows a count. The usable images can still continue.

## 2. Check source and license

Record where the images came from and the permission that allows you to use them. Sprite Lab pauses safely when either item is missing. Do not continue with images you are not allowed to use.

Saved pack information remains editable and is prefilled when you reopen it. Returning to the Dataset page keeps the approved folder selected, so you can continue the same build without choosing it again.

## 3. Build the dataset

To include hierarchical taxonomy suggestions, open **Settings**, enable **Hierarchical labeling**, choose a processing
profile and a reference cohort size from 300 through 500, then save. The project-scoped setting is applied immediately
to labeling status and to future dataset builds. Enabling it does not retroactively process an existing dataset, so
rebuild that dataset when hierarchical artifacts are required.

The **Labeling** page can prepare those artifacts directly. **Prepare labeling** runs the managed dataset pass in a
background thread and shows stage progress plus timestamped activity logs; navigating around the page does not block
the job. When preparation completes, the semantic review queue refreshes automatically. Every eligible image is
prefilled first. A valid proposal at or above `0.8` confidence, without conflicts or provider-health warnings, is kept
as an automatic prefill and does not require human truth. Lower-certainty proposals and provider abstentions are
excluded from semantic supervision by default. They appear in an optional rescue queue only when a user wants to keep
their semantic label; saving that rescue requires an append-only human review event. The image remains available to
the image-only dataset even when its semantic label is excluded.

Choose **Build dataset**. Sprite Lab checks copies of the images and reports how many are ready, excluded, or need attention.

When the build finishes, use the displayed review actions to open exclusion rescue or image-description review directly.

The Dataset page remembers the active source-folder approval for the current application session and restores a
project-selected imported dataset across restarts. Back buttons connect Dataset, pack information, shared review,
exception review, and labeling without requiring another folder choice.

Choose **Review whole dataset** to audit every manifest image. The review toolbar can show default exceptions,
potentially problematic accepted images, all accepted images, all excluded images, or the complete dataset, and can
filter by reason or filename. Excluding an accepted image records a human review decision and republishes only managed
derived artifacts; it does not change the source image folder.

An optional **Rescue images** page may appear:

> Sprite Lab excluded these images automatically.
>
> You only need to select images that should be kept. Everything else can remain excluded.

You can rescue selected images or continue without reviewing them. Skipping this optional review keeps the automatic exclusions unchanged.

If a low-certainty description should be retained for semantic training, choose **Review images** and record the best
defensible description. Unreviewed low-certainty descriptions stay excluded; they do not block image-only training.

Next: [Train a model](train_a_model.md).
