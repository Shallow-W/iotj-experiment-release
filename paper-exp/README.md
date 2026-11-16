# Original manuscript experiments

This directory accompanies Section V of *A Two-Plane Hierarchical Framework for Scalable Joint Microservice Deployment and Request Routing in Dynamic Internet of Vehicles*. It contains simulator inputs, historical training workbooks, the Python training code, and the artwork embedded in the original manuscript.

## Figure and result index

The figure identifiers below follow the TeX artwork filenames.

| Experiment | Manuscript artwork | Available data and code |
|---|---|---|
| Service-cluster partitioning | [10a–c](figures/clustering/) | The original cluster assignments and figure source tables are unavailable. [Capacity-constrained K-means prototype](code/clustering/balanced_kmeans.py). The revision includes a separate [FPDWGC and clustering evaluation](../reviewer_3/02_r3_2_clustering_baselines/). |
| Priority experience replay ablation | [11a–b](figures/per/) | [Saved training workbooks](results/training/) for the full model and d3qn_mask_nopri. |
| Feasibility mask ablation | [12a–b](figures/masking/) | [Saved training workbooks](results/training/) for the full model and d3qn_nomask_pri. |
| End-to-end delay by system scale | [13a–d](figures/delay_by_scale/) | [Historical plotting script](code/archived_plots/draw2.py) and [50- and 100-node runner variants](code/training/). Several bar heights are computed from fixed multipliers in the plotting script. Source run logs for the full sweep are unavailable. |
| Load variance by system scale | [14a–d](figures/load_variance_by_scale/) | [Historical plotting script](code/archived_plots/draw2.py) and a [saved resource snapshot](results/training/resources.xlsx). The full measured sweep could not be traced to run logs. |
| Training reward across methods | [15](figures/training_reward/fig15.png) | [Saved training workbooks](results/training/) for MPD3QN, DQN, SAC and GDDPG. The [historical plotting script](code/archived_plots/draw3_copy.py) rescales baseline rewards and generates the shading procedurally. The bands do not estimate uncertainty across repeated training runs. |

The historical workbook snapshots preserve 1,000 episode rows per delay/reward file. Reward columns contain step-level values within each episode, and each delay row contains the saved final delay. The training code includes closely related MPD3QN versions. The exact seed and code revision for each saved workbook were not recorded with the result.

The original ablation drawing script uses smoothed returns and a running minimum for delay. It also contains comparison-label and display-axis choices that differ from the saved data. The script is retained in [archived_plots](code/archived_plots/) to document those choices.

The clustering prototype uses scikit-learn 1.6.1. Its standalone example generates synthetic points.

## Plot the saved training records

Run from the repository root with Python 3.9.

~~~bash
python3 -m pip install -r paper-exp/requirements.txt
python3 paper-exp/code/plotting/plot_saved_training.py
~~~

The script sums per-step rewards for each episode and plots final delays as recorded. It writes the PER ablation, masking ablation and reward comparison to paper-exp/reproduced/figures/. These plots use the workbook values directly. Historical manuscript artwork is under [figures](figures/).

## Run training

~~~bash
python3 -m pip install -r paper-exp/requirements-training.txt
MPLBACKEND=Agg IOTJ_EPISODES=1 python3 paper-exp/code/training/D3QN_STATEMASK_PRI_YANMA.py
~~~

The one-episode run checks that training and result export work. Omit IOTJ_EPISODES to use the original 1,000-episode setting. Outputs go to a new directory under paper-exp/reproduced/training/.

| Method or variant | Script |
|---|---|
| MPD3QN with feasibility masking and PER | [D3QN_STATEMASK_PRI_YANMA.py](code/training/D3QN_STATEMASK_PRI_YANMA.py) |
| MPD3QN variant with the saved full-model output filename | [D3QN_STATEMASK_PRI_xiugai.py](code/training/D3QN_STATEMASK_PRI_xiugai.py) |
| D3QN variants for ablations | [Without PER](code/training/D3QN_NoPRI.py), [without masking](code/training/D3QN_NoStateMask.py), [without either](code/training/D3QN_NOmask_NoPRI.py) |
| Baselines | [DQN](code/training/DQN.py), [SAC](code/training/SAC.py), [GDDPG](code/training/DDPG.py) |
| Scale variants | [50 MEC nodes](code/training/50nodes_D3QN_PRI.py), [100 MEC nodes](code/training/100nodes_D3QN_PRI.py) |

The training entries use the [50- and 100-endpoint simulator inputs](inputs/), including aggregated request counts derived for the simulation. The underlying PeMS03 detector archive is a separate data source. The revision runners use a different app.xlsx in each scenario. See the [supplementary experiment instructions](../EXPERIMENTS.md) for the exact mapping.

The release copies replace the original machine-specific paths with package-relative inputs and outputs. They also correct a tuple subtraction in the GDDPG reward update. The 100-node runner now infers the network input size from the masked state and converts the resource state to an array before export.
