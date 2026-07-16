"""Centralized, reusable product language for the novice experience."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class ProductCopy:
    """User-facing copy for one explicit product presentation state.

    Feature code remains responsible for deciding which state applies. This
    catalog only supplies language and actions for a state selected elsewhere.
    """

    key: str
    title: str
    body: tuple[str, ...]
    primary_action: str
    next_action: str
    technical_details: str | None = None
    details_label: str = "Technical details"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def plain_text(self) -> str:
        """Return the always-visible language, excluding expandable details."""

        return "\n".join((self.title, *self.body, self.primary_action, self.next_action))


_ENTRIES = (
    ProductCopy(
        key="welcome",
        title="Welcome to Sprite Lab",
        body=(
            "Turn a folder of images into a model you can try.",
            "Sprite Lab guides you through each check and stops before anything costly begins.",
        ),
        primary_action="Choose image folder",
        next_action="Choose the folder that contains the images you want to use.",
    ),
    ProductCopy(
        key="first_dataset",
        title="Build your first dataset",
        body=(
            "A dataset is the collection of images Sprite Lab will prepare for learning.",
            "Your original files stay where they are.",
        ),
        primary_action="Check this folder",
        next_action="Choose Check this folder to confirm the image source and license.",
    ),
    ProductCopy(
        key="missing_source",
        title="Tell us where the images came from",
        body=("Sprite Lab needs a source so you can keep a clear record of the images you use.",),
        primary_action="Add image source",
        next_action="Enter the website, collection, artist, or other source shown with the images.",
        technical_details="The image-source field is empty for this collection.",
    ),
    ProductCopy(
        key="missing_license",
        title="Add license information",
        body=(
            "Sprite Lab could not find permission information for these images.",
            "Do not continue until you know you are allowed to use them.",
        ),
        primary_action="Add license information",
        next_action="Find the license or permission note that came with the images, then add it here.",
        technical_details="No license record was supplied for this image collection.",
    ),
    ProductCopy(
        key="preprocessing_summary",
        title="Image check complete",
        body=(
            "Images Sprite Lab could read are ready.",
            "Damaged or unsupported files stayed out and your originals were not changed.",
        ),
        primary_action="Continue",
        next_action="Review the image counts, then choose Continue.",
    ),
    ProductCopy(
        key="optional_review",
        title="Rescue images",
        body=(
            "Sprite Lab excluded these images automatically.",
            "You only need to select images that should be kept. Everything else can remain excluded.",
        ),
        primary_action="Rescue images",
        next_action="Choose Rescue images, or continue without reviewing them now.",
        technical_details="This optional review only changes which automatically excluded images are kept.",
    ),
    ProductCopy(
        key="training_blocked",
        title="Training is temporarily unavailable",
        body=(
            "Training is temporarily unavailable because its safety checks need repair.",
            "Your dataset and completed work are safe.",
        ),
        primary_action="Show what to fix",
        next_action="Choose Show what to fix and complete the listed repair before training.",
        technical_details="A required training safety check is not currently passing.",
    ),
    ProductCopy(
        key="training_confirmation",
        title="Ready to train",
        body=("Training can take time and use a lot of your computer's processing power.",),
        primary_action="Start training",
        next_action="Review the time and computer-use estimate, then choose Start training once.",
    ),
    ProductCopy(
        key="hosted_cost_confirmation",
        title="Confirm hosted training cost",
        body=(
            "Training on a hosted computer may create a charge.",
            "Nothing will start until you confirm the shown provider, limit, and estimated cost.",
        ),
        primary_action="Confirm and start hosted training",
        next_action="Check the cost limit, then confirm once to start hosted training.",
    ),
    ProductCopy(
        key="evaluation",
        title="See how your model is doing",
        body=("Try familiar prompts, compare the results, and keep notes about what works.",),
        primary_action="Evaluate model",
        next_action="Choose Evaluate model, then try a prompt in the playground.",
    ),
    ProductCopy(
        key="memorization_block",
        title="This model needs more work",
        body=(
            "Sprite Lab found generated images that are too close to images used for training.",
            "The model will stay blocked from release.",
        ),
        primary_action="Review image pairs",
        next_action="Review the flagged image pairs, then train again before release.",
        technical_details="The similarity safety check blocked this model result.",
    ),
    ProductCopy(
        key="review_required",
        title="A quick image review is needed",
        body=("Some images still need a decision before the next step can begin.",),
        primary_action="Review images",
        next_action="Open Review images and choose the best description for each flagged image.",
    ),
    ProductCopy(
        key="unexpected_error",
        title="Sprite Lab ran into a problem",
        body=(
            "Your completed work is safe.",
            "Private technical details stay hidden unless you choose to include them in a support report.",
        ),
        primary_action="Try again",
        next_action="Choose Try again. If the problem returns, open Troubleshooting from Help.",
        technical_details="A private diagnostic reference is available for support; no error trace is shown here.",
    ),
    ProductCopy(
        key="interrupted_run",
        title="The run was interrupted",
        body=("The run stopped before it finished. Completed work was saved.",),
        primary_action="Check saved work",
        next_action="Choose Check saved work before continuing the run.",
    ),
    ProductCopy(
        key="safe_resume",
        title="Ready to continue",
        body=("Sprite Lab checked the saved work and can continue safely.",),
        primary_action="Continue run",
        next_action="Choose Continue run to pick up from the last safe stopping point.",
        technical_details="The current project still matches the saved run.",
    ),
    ProductCopy(
        key="no_data",
        title="No usable images found",
        body=("This folder did not contain images Sprite Lab can prepare.",),
        primary_action="Choose another folder",
        next_action="Choose another folder that contains PNG, JPEG, or WebP images.",
    ),
    ProductCopy(
        key="no_provider",
        title="Image descriptions are not set up",
        body=(
            "Sprite Lab needs an image description service to suggest labels automatically.",
            "You can connect a local or hosted option, or add descriptions yourself.",
        ),
        primary_action="Choose a description option",
        next_action="Choose a local option, connect a hosted option, or continue with manual descriptions.",
        technical_details="No vision service is currently available.",
    ),
    ProductCopy(
        key="provider_authentication",
        title="Sign-in needs attention",
        body=(
            "Sprite Lab could not sign in to the image description service.",
            "Your sign-in details were not saved in the project.",
        ),
        primary_action="Open connection settings",
        next_action="Open connection settings, sign in again, then choose Test connection.",
        technical_details="The remote service rejected the current sign-in attempt.",
    ),
    ProductCopy(
        key="remote_connection_loss",
        title="Connection lost",
        body=(
            "Sprite Lab lost its connection to the remote service.",
            "Completed work was saved and no new remote work will start automatically.",
        ),
        primary_action="Try connection again",
        next_action="Check your internet connection, then choose Try connection again.",
    ),
)

COPY_CATALOG = MappingProxyType({entry.key: entry for entry in _ENTRIES})


def copy_for(key: str) -> ProductCopy:
    """Return copy for an explicit presentation key selected by feature code."""

    try:
        return COPY_CATALOG[key]
    except KeyError:
        raise KeyError(f"Unknown product copy key: {key}") from None


def copy_catalog_payload() -> dict[str, Any]:
    """Return a JSON-ready catalog in stable key order."""

    return {
        "schema_version": "spritelab.product-copy.v1",
        "principles": [
            "Use plain language before technical detail.",
            "Say what happened, what was preserved, and exactly what to do next.",
            "Keep technical detail optional and expandable.",
            "Never ask an ordinary user for an internal file location or fingerprint.",
            "Confirm an expensive operation once, immediately before it starts.",
        ],
        "entries": [entry.to_dict() for entry in _ENTRIES],
    }
