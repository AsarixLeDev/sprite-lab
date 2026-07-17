# Evaluate and generate

Evaluation helps you see what the model does well and where it needs more work.

Choose **Evaluate model** to run the prepared examples. Review the results for image quality and whether they match the requested subject.

Then open the prompt playground:

1. Enter a short description, such as “a blue potion with a silver cap.”
2. Choose **Try a prompt**.
3. Compare the image with your request and save useful notes.

## Exploratory infrastructure smoke checkpoints

When a completed conditioned Dataset-v5 publication is waiting for activation, the Playground can register its two-step infrastructure smoke checkpoint. Open **Register a 2-step infrastructure smoke checkpoint**, select the eligible conditioned publication, and choose **Prepare smoke plan**. Sprite Lab derives the candidate, publication, freeze, campaign, and training-code identities on the server; no hashes or file paths need to be copied into the browser.

Choose **Run CPU smoke** first. After it completes, choose **Run CUDA smoke**. Sprite Lab launches only the fixed server-prepared argument list, with the required environment set before Python starts. The page shows durable status and privacy-safe log tails. It never starts either 5,000-step campaign output and never changes `spritelab.yaml`.

After both completion receipts are verified, choose **Register for Playground only**. Receipt identities are read by the server, not pasted by the user. The CUDA live and EMA step-2 checkpoints are copied into the separate exploratory checkpoint catalog. They are always labeled exploratory and are never eligible for production Evaluation, training resume, promotion, or campaign-execution evidence.

A failed or interrupted device smoke cannot resume. Choose **Use fresh retry nonce**, prepare a new bundle, and run CPU then CUDA again. Page load and status display do not import Torch, initialize CUDA, launch a process, or create smoke directories.

If Sprite Lab finds a generated image that is too close to an image used for training, the model stays blocked from release. Choose **Review image pairs**, then adjust the dataset or training and try again.

No evaluation result releases a model automatically.
