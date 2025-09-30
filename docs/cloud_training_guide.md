# Cloud Training Guide

This guide explains how to train the Texas Hold'em AI on Google Colab and on
Google Cloud Platform (GCP).  Each workflow includes prerequisites, environment
setup, and the commands required to launch training runs using the
`poker_ai.cli.train` entry point.

---

## 1. Google Colab Workflow

### 1.1 Launch a GPU-backed notebook
1. Open [Google Colab](https://colab.research.google.com/).
2. Click **File → New Notebook**.
3. Enable a GPU runtime via **Runtime → Change runtime type → Hardware
   accelerator → GPU → Save**.

> 💡 TPU runtimes are also supported.  Select **TPU** instead of **GPU** and
> follow the TPU-specific steps in [section 1.4](#14-optional-tpu-runtime).

### 1.2 Install dependencies
Run the following cell to install system packages and Python dependencies:

```python
!sudo apt-get update -y
!sudo apt-get install -y git
!git clone https://github.com/OWNER/texas_holdem_ai_gatc.git
%cd texas_holdem_ai_gatc
!pip install -r requirements.txt
```

If you plan to store checkpoints on Google Drive, mount it with:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Update the training scripts to save checkpoints to the mounted path (e.g.
`--output-dir /content/drive/MyDrive/poker_ai_runs`).

### 1.3 Run a baseline training session
Use the high-level wrapper to start training.  The example below runs a short
Deep CFR session on the GPU runtime:

```python
!python -m poker_ai.cli.train \
    --algorithm deep_cfr \
    --num-hands 5000 \
    --batch-size 2048 \
    --gpus
```

Key flags:

- `--algorithm`: Selects the trainer (`deep_cfr`, `mccfr`, etc.).
- `--num-hands`: Number of self-play hands to simulate.
- `--gpus`: Enables GPU acceleration (all visible GPUs are used).
- `--output-dir`: Optional directory for checkpoints and logs.
- `--save-model-every`: Checkpoint frequency in number of hands.

Monitor logs directly in the notebook output.  Training creates a `runs/`
subdirectory with TensorBoard summaries.  To visualize metrics:

```python
%load_ext tensorboard
%tensorboard --logdir runs
```

### 1.4 Optional TPU runtime
If you selected a TPU runtime in Colab:

```python
!pip install torch==2.2.0 torch-xla==2.2.0 torchvision==0.17.0 -f \
    https://storage.googleapis.com/tpu-pytorch/wheels/colab.html
```

Start training with TPU support:

```python
!python -m poker_ai.cli.train \
    --algorithm deep_cfr \
    --num-hands 5000 \
    --tpu
```

The trainer automatically falls back to CPU for evaluation phases while TPU
kernels run self-play and optimization.

---

## 2. Google Cloud Platform Workflow

### 2.1 Prerequisites
- A Google Cloud project with billing enabled.
- The [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed
  locally.
- Quota for the desired accelerator (GPU or TPU).

Authenticate with your account:

```bash
gcloud auth login
gcloud auth application-default login
```

### 2.2 Provision a GPU VM
1. Choose a region and zone with available GPUs (e.g. `us-central1` /
   `us-central1-a`).
2. Create the VM using the helper script bundled with the project:

   ```bash
   poker-ai-gcp-train \
     --project <PROJECT_ID> \
     --zone us-central1-a \
     --name poker-ai-gpu \
     --accelerator gpu \
     --machine-type n1-standard-8 \
     create --gpu-type nvidia-tesla-t4
   ```

   The wrapper provisions a Compute Engine VM with the selected GPU and
   installs CUDA drivers.

3. Connect to the VM:

   ```bash
   gcloud compute ssh poker-ai-gpu --zone us-central1-a
   ```

4. Inside the VM, clone the repository and install dependencies:

   ```bash
   git clone https://github.com/OWNER/texas_holdem_ai_gatc.git
   cd texas_holdem_ai_gatc
   pip install -r requirements.txt
   ```

5. Launch a GPU training job:

   ```bash
   python -m poker_ai.cli.train \
       --algorithm deep_cfr \
       --num-hands 100000 \
       --gpus \
       --save-model-every 5000 \
       --output-dir /home/$USER/poker_runs
   ```

6. (Optional) Stream TensorBoard logs to your local machine:

   ```bash
   gcloud compute scp \
     --recurse \
     poker-ai-gpu:/home/$USER/poker_runs \
     ./poker_runs --zone us-central1-a
   tensorboard --logdir ./poker_runs
   ```

### 2.3 Provision a TPU VM
1. Request TPU quota in the desired region.
2. Create a TPU VM using the helper:

   ```bash
   poker-ai-gcp-train \
     --project <PROJECT_ID> \
     --zone us-central2-b \
     --name poker-ai-tpu \
     --accelerator tpu \
     create --tpu-type v4-8
   ```

3. SSH into the TPU VM:

   ```bash
   gcloud alpha compute tpus tpu-vm ssh poker-ai-tpu --zone us-central2-b
   ```

4. Install TPU-compatible PyTorch wheels and project dependencies:

   ```bash
   pip install torch==2.2.0 torch-xla==2.2.0 torchvision==0.17.0 -f \
       https://storage.googleapis.com/tpu-pytorch/wheels/colab.html
   git clone https://github.com/OWNER/texas_holdem_ai_gatc.git
   cd texas_holdem_ai_gatc
   pip install -r requirements.txt
   ```

5. Start TPU-enabled training:

   ```bash
   python -m poker_ai.cli.train \
       --algorithm deep_cfr \
       --num-hands 200000 \
       --tpu \
       --save-model-every 10000 \
       --output-dir /home/$USER/poker_runs
   ```

### 2.4 Automating multiple runs
The helper supports starting, listing, and deleting resources.  Examples:

```bash
# List active workers
poker-ai-gcp-train --project <PROJECT_ID> --zone us-central1-a list

# Delete the GPU VM when finished
gcloud compute instances delete poker-ai-gpu --zone us-central1-a

# Delete the TPU VM
gcloud alpha compute tpus tpu-vm delete poker-ai-tpu --zone us-central2-b
```

Automate experiments by scripting repeated invocations of `poker-ai-gcp-train`
with different flags or by using Cloud Scheduler to trigger custom startup
scripts stored in Cloud Storage.

---

## 3. Tips and Best Practices
- **Checkpointing**: Use `--save-model-every` and `--output-dir` to persist
  models.  Store artifacts on durable storage (`/content/drive` in Colab or
  Google Cloud Storage buckets on GCP).
- **Monitoring**: Enable TensorBoard or write logs to Cloud Logging when running
  unattended jobs.
- **Cost control**: Stop or delete VMs when idle to avoid unexpected charges.
- **Reproducibility**: Pin critical hyperparameters in configuration files or
  pass them explicitly on the command line.  Save the Git commit hash alongside
  checkpoints for future audits.

Following the steps above ensures a reliable, reproducible training workflow on
both Colab and Google Cloud.
