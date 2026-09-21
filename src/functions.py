'''
This module contains functions used to visualize and analyze the results of models' estimation. 

Functions:
plot_collinearity_summary() - visualizes the multicollinearity metrics (VIF and CI) for explanatory variables, 
plot_bias() - calculates and visualizes the bias of the estimators, 
plot_sd() - calculates and visualizes the standar deviation of the estimators, 
plot_mse() - calculates and visualizes the mean squared error of the estimators, 
plot_sign_accuracy() - visualizes the accuracy of the signs of the estimators, 
plot_metrics_heatmap() - visualizes the R2 and cross-validation MSE of the models.


wrapper_bias(), wrapper_sd(), wrapper_mse(), wrapper_sign_accuracy(), wrapper_heatmap()
- wrapper functions, allowing to plot interactive visualizations for different analysis scenarios

'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ipywidgets import interact, IntSlider, Dropdown

np.random.seed(42)

# plot_collinearity_summary

def plot_collinearity_summary(wyniki, scenarios, sample_sizes, rhos):

    fig, axes = plt.subplots(1, len(scenarios),
                             figsize=(12, 5),
                             sharey=True)

    if len(scenarios) == 1:
        axes = [axes]

    color_map = {
        ("vif", 50): "#9D4EDD",  
        ("vif", 500): "#240046",
        ("ci", 50): "#FF9E00",      
        ("ci", 500): "#FF6D00"
    }

    scenario_labels = {
        "baseline": "Scenario 1 – baseline",
        "extra_vars": "Scenario 2 – extra variables",
        "omitted_vars": "Scenario 3 – omitted variables"
    }

    for k, scenario in enumerate(scenarios):

        ax = axes[k]
        line_dict = {}

        for sample_size in sample_sizes:

            vif_vals = []
            ci_vals = []

            for rho in rhos:

                wybrane = [
                    w for w in wyniki
                    if w["rho"] == rho
                    and w["sample_size"] == sample_size
                    and w["scenario"] == scenario
                ]

                vif_vals.append(np.mean([w["vif_mean"] for w in wybrane]))
                ci_vals.append(np.mean([w["ci_max"] for w in wybrane]))

            # VIF
            line_vif, = ax.plot(
                rhos,
                vif_vals,
                marker='o',
                linestyle='-',
                color=color_map[("vif", sample_size)],
                label=f"VIF n={sample_size}"
            )

            # CI
            line_ci, = ax.plot(
                rhos,
                ci_vals,
                marker='o',
                linestyle='-',
                color=color_map[("ci", sample_size)],
                label=f"CI n={sample_size}"
            )

            line_dict[("vif", sample_size)] = line_vif
            line_dict[("ci", sample_size)] = line_ci

        ax.set_title(scenario_labels[scenario])
        ax.set_xlabel("ρ")
        ax.set_xticks(rhos)
        
        ax.set_yticks([1, 5, 10, 15, 20])

        if k == 0:
            ax.set_ylabel("")

        order = [
            ("vif", 50),
            ("vif", 500),
            ("ci", 50),
            ("ci", 500)
        ]

        handles = [line_dict[o] for o in order]
        labels = [
            r"$\overline{\mathrm{VIF}}$, n=50",
            r"$\overline{\mathrm{VIF}}$, n=500",
            r"$\overline{\mathrm{CI}}_{max}$, n=50",
            r"$\overline{\mathrm{CI}}_{max}$, n=500"
        ]

        ax.legend(handles, labels)

    plt.tight_layout()
    plt.show()


# plot_bias

def plot_bias(wyniki, beta_idx, scenario_choice):

    fig, axes = plt.subplots(2, 2, figsize=(12, 5.4), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, rho in enumerate(sorted(set(w["rho"] for w in wyniki))):

        ax = axes[i]

        for sample_size in sorted(set(w["sample_size"] for w in wyniki)):

            wybrane = [
                w for w in wyniki
                if w["sample_size"] == sample_size
                and w["rho"] == rho
                and w["scenario"] == scenario_choice
            ]

            if not wybrane:
                continue

            wybrane = sorted(wybrane, key=lambda w: w["components_ratio"])
            ratio_list = [w["components_ratio"] for w in wybrane]

            bias_pca = []
            bias_pls = []

            betas_true = np.array(wybrane[0]["betas_true"])

            for w in wybrane:
                pca_params = np.array(w["params_x_pca"])
                pls_params = np.array(w["params_x_pls"])

                pca_mean = pca_params.mean(axis=0)
                pls_mean = pls_params.mean(axis=0)

                bias_pca.append(abs(pca_mean[beta_idx] - betas_true[beta_idx]))
                bias_pls.append(abs(pls_mean[beta_idx] - betas_true[beta_idx]))


            key = (scenario_choice, tuple(betas_true))

            if key == ("baseline", (1,1,1,1,1)):
                if sample_size == 50:
                    color_pca = "#F08A9B"
                    color_pls = "#66BFBF"
                else:
                    color_pca = "#DC143C"
                    color_pls = "#008B8B"

            elif key == ("baseline", (5,0.5,0.5,0.5,0.5)):
                if sample_size == 50:
                    color_pca = "#6FBF6F"
                    color_pls = "#FF66FF"
                else:
                    color_pca = "#228B22"
                    color_pls = "#FF00FF"

            elif key == ("baseline", (3,-3,3,-3,3)):
                if sample_size == 50:
                    color_pca = "#FFD966"
                    color_pls = "#8FBCE6"
                else:
                    color_pca = "#FFD700"
                    color_pls = "#6495ED"

            elif key == ("baseline", (1,2,3,4,5)):
                if sample_size == 50:
                    color_pca = "#FF85C2"
                    color_pls = "#FFB347"
                else:
                    color_pca = "#FF1493"
                    color_pls = "#FFA500"

            elif key == ("extra_vars", (1,2,3,4,5)):
                if sample_size == 50:
                    color_pca = "#C8A2C8"
                    color_pls = "seagreen"
                else:
                    color_pca = "#800080"
                    color_pls = "darkgreen"

            elif key == ("omitted_vars", (1,2,3,4,5)):
                if sample_size == 50:
                    color_pca = "#C49A7A"
                    color_pls = "slateblue"
                else:
                    color_pca = "#8B4513"
                    color_pls = "darkblue"

            else:
                color_pca = "gray"
                color_pls = "gray"

            ax.plot(
                ratio_list,
                bias_pca,
                marker='o',
                linestyle='-',
                color=color_pca,
                label=f"PCR, n={sample_size}"
            )

            ax.plot(
                ratio_list,
                bias_pls,
                marker='o',
                linestyle='-',
                color=color_pls,
                label=f"PLSR, n={sample_size}"
            )


        for sample_size in sorted(set(w["sample_size"] for w in wyniki)):

            wybrane_ols = [
                w for w in wyniki
                if w["rho"] == rho
                and w["scenario"] == scenario_choice
                and w["sample_size"] == sample_size
            ]

            if not wybrane_ols:
                continue

            ols_bias_vals = []

            for w in wybrane_ols:
                ols_params = np.array(w["params_x_ols"])
                betas_true = np.array(w["betas_true"])

                ols_mean = ols_params.mean(axis=0)
                ols_bias_vals.append(ols_mean[beta_idx] - betas_true[beta_idx])

            ols_bias = abs(np.mean(ols_bias_vals))

            if sample_size == 50:
                color_mnk = "#999999"   
            else:
                color_mnk = "#000000"   

            ax.axhline(
                ols_bias,
                linestyle=':',
                linewidth=2,
                color=color_mnk,
                label=f"MNK, n={sample_size}"
            )

        ax.set_title(f"ρ = {rho}")
        ax.set_xlabel("Explained variance (PCA)")
        ax.set_ylabel("|Bias|")

        ax.set_xticks(ratio_list)
        ax.set_xticklabels([f"{x:.1f}" for x in ratio_list])
        ax.tick_params(labelbottom=True)


        if i == 0:
            handles, labels = ax.get_legend_handles_labels()

            order_labels = [
                "PCR, n=50",
                "PCR, n=500",
                "PLSR, n=50",
                "PLSR, n=500",
                "MNK, n=50",
                "MNK, n=500"
            ]

            ordered_handles = []
            ordered_labels = []

            for lab in order_labels:
                for h, l in zip(handles, labels):
                    if l == lab:
                        ordered_handles.append(h)
                        ordered_labels.append(l)

            ax.legend(ordered_handles, ordered_labels)

    plt.tight_layout()
    plt.show()

# wrapper_bias

def wrapper_bias(wyniki):

    scenario_dropdown = Dropdown(
        options=sorted(list(set(w["scenario"] for w in wyniki))),
        description="scenario"
    )

    beta_slider = IntSlider(min=0, max=4, step=1, value=0, description="beta_idx")

    betas_dropdown = Dropdown(description="betas")

    def to_tuple(b):
        return tuple(float(x) for x in b)

    def get_betas_for_scenario(scenario):
        betas = {
            to_tuple(w["betas_true"])
            for w in wyniki
            if w["scenario"] == scenario
        }
        return {str(b): b for b in sorted(betas)}

    def update(*args):
        scenario = scenario_dropdown.value
        options = get_betas_for_scenario(scenario)

        betas_dropdown.options = options

        if options:
            betas_dropdown.value = list(options.values())[0]

    scenario_dropdown.observe(update, names='value')
    update()

    def plot_for_selection(scenario_choice, beta_idx, betas_choice):

        if betas_choice is None:
            return

        filt = [
            w for w in wyniki
            if w["scenario"] == scenario_choice
            and to_tuple(w["betas_true"]) == to_tuple(betas_choice)
        ]

        plot_bias(filt, beta_idx, scenario_choice)

    interact(
        plot_for_selection,
        scenario_choice=scenario_dropdown,
        beta_idx=beta_slider,
        betas_choice=betas_dropdown
    )

# plot_sd

def plot_sd(wyniki, beta_idx, scenario_choice):

    fig, axes = plt.subplots(2, 2, figsize=(12, 5.4), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, rho in enumerate(sorted(set(w["rho"] for w in wyniki))):

        ax = axes[i]


        for sample_size in sorted(set(w["sample_size"] for w in wyniki)):

            wybrane = [
                w for w in wyniki
                if w["sample_size"] == sample_size
                and w["rho"] == rho
                and w["scenario"] == scenario_choice
            ]

            if not wybrane:
                continue

            wybrane = sorted(wybrane, key=lambda w: w["components_ratio"])
            ratio_list = [w["components_ratio"] for w in wybrane]

            sd_pca = []
            sd_pls = []

            betas_true = np.array(wybrane[0]["betas_true"])

            for w in wybrane:

                pca_params = np.array(w["params_x_pca"])
                pls_params = np.array(w["params_x_pls"])

                sd_pca.append(np.std(pca_params[:, beta_idx]))
                sd_pls.append(np.std(pls_params[:, beta_idx]))

            key = (scenario_choice, tuple(betas_true))

            if key == ("baseline", (1,1,1,1,1)):
                if sample_size == 50:
                    color_pca = "#F08A9B"
                    color_pls = "#66BFBF"
                else:
                    color_pca = "#DC143C"
                    color_pls = "#008B8B"

            elif key == ("baseline", (5,0.5,0.5,0.5,0.5)):
                if sample_size == 50:
                    color_pca = "#6FBF6F"
                    color_pls = "#FF66FF"
                else:
                    color_pca = "#228B22"
                    color_pls = "#FF00FF"

            elif key == ("baseline", (3,-3,3,-3,3)):
                if sample_size == 50:
                    color_pca = "#FFD966"
                    color_pls = "#8FBCE6"
                else:
                    color_pca = "#FFD700"
                    color_pls = "#6495ED"

            elif key == ("baseline", (1,2,3,4,5)):
                if sample_size == 50:
                    color_pca = "#FF85C2"
                    color_pls = "#FFB347"
                else:
                    color_pca = "#FF1493"
                    color_pls = "#FFA500"

            elif key == ("extra_vars", (1,2,3,4,5)):
                if sample_size == 50:
                    color_pca = "#C8A2C8"
                    color_pls = "seagreen"
                else:
                    color_pca = "#800080"
                    color_pls = "darkgreen"

            elif key == ("omitted_vars", (1,2,3,4,5)):
                if sample_size == 50:
                    color_pca = "#C49A7A"
                    color_pls = "slateblue"
                else:
                    color_pca = "#8B4513"
                    color_pls = "darkblue"

            else:
                color_pca = "gray"
                color_pls = "gray"


            ax.plot(
                ratio_list,
                sd_pca,
                marker='o',
                linestyle='-',
                color=color_pca,
                label=f"PCR, n={sample_size}"
            )

            ax.plot(
                ratio_list,
                sd_pls,
                marker='o',
                linestyle='-',
                color=color_pls,
                label=f"PLSR, n={sample_size}"
            )

        for sample_size in sorted(set(w["sample_size"] for w in wyniki)):

            wybrane_ols = [
                w for w in wyniki
                if w["rho"] == rho
                and w["scenario"] == scenario_choice
                and w["sample_size"] == sample_size
            ]

            if not wybrane_ols:
                continue

            sd_vals = []

            for w in wybrane_ols:
                ols_params = np.array(w["params_x_ols"])
                sd_vals.append(np.std(ols_params[:, beta_idx]))

            sd_ols = np.mean(sd_vals)

            if sample_size == 50:
                color_mnk = "#999999"
            else:
                color_mnk = "#000000"

            ax.axhline(
                sd_ols,
                linestyle=':',
                linewidth=2,
                color=color_mnk,
                label=f"MNK, n={sample_size}"
            )


        ax.set_title(f"ρ = {rho}")
        ax.set_xlabel("Explained variance (PCA)")
        ax.set_ylabel("SD(β̂)")

        ax.set_xticks(ratio_list)
        ax.set_xticklabels([f"{x:.1f}" for x in ratio_list])
        ax.tick_params(labelbottom=True)

        if i == 0:
            handles, labels = ax.get_legend_handles_labels()

            order_labels = [
                "PCR, n=50",
                "PCR, n=500",
                "PLSR, n=50",
                "PLSR, n=500",
                "MNK, n=50",
                "MNK, n=500"
            ]

            ordered_handles = []
            ordered_labels = []

            for lab in order_labels:
                for h, l in zip(handles, labels):
                    if l == lab:
                        ordered_handles.append(h)
                        ordered_labels.append(l)

            ax.legend(ordered_handles, ordered_labels)

    plt.tight_layout()
    plt.show()

# wrapper_sd

def wrapper_sd(wyniki):

    scenario_dropdown = Dropdown(
        options=sorted(list(set(w["scenario"] for w in wyniki))),
        description="scenario"
    )

    beta_slider = IntSlider(min=0, max=4, step=1, value=0, description="beta_idx")

    betas_dropdown = Dropdown(description="betas")

    def to_tuple(b):
        return tuple(float(x) for x in b)

    def get_betas_for_scenario(scenario):
        betas = {
            to_tuple(w["betas_true"])
            for w in wyniki
            if w["scenario"] == scenario
        }
        return {str(b): b for b in sorted(betas)}

    def update(*args):
        scenario = scenario_dropdown.value
        options = get_betas_for_scenario(scenario)

        betas_dropdown.options = options

        if options:
            betas_dropdown.value = list(options.values())[0]

    scenario_dropdown.observe(update, names='value')
    update()

    def plot_for_selection(scenario_choice, beta_idx, betas_choice):

        if betas_choice is None:
            return

        filt = [
            w for w in wyniki
            if w["scenario"] == scenario_choice
            and to_tuple(w["betas_true"]) == to_tuple(betas_choice)
        ]

        plot_sd(filt, beta_idx, scenario_choice)

    interact(
        plot_for_selection,
        scenario_choice=scenario_dropdown,
        beta_idx=beta_slider,
        betas_choice=betas_dropdown
    )

# plot_mse

def plot_mse(wyniki, beta_idx, scenario_choice):

    fig, axes = plt.subplots(2, 2, figsize=(12, 5.4), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, rho in enumerate(sorted(set(w["rho"] for w in wyniki))):

        ax = axes[i]


        for sample_size in sorted(set(w["sample_size"] for w in wyniki)):

            wybrane = [
                w for w in wyniki
                if w["sample_size"] == sample_size
                and w["rho"] == rho
                and w["scenario"] == scenario_choice
            ]

            if not wybrane:
                continue

            wybrane = sorted(wybrane, key=lambda w: w["components_ratio"])
            ratio_list = [w["components_ratio"] for w in wybrane]

            mse_pca = []
            mse_pls = []

            betas_true = np.array(wybrane[0]["betas_true"])

            for w in wybrane:

                pca_params = np.array(w["params_x_pca"])
                pls_params = np.array(w["params_x_pls"])

                if beta_idx < pca_params.shape[1]:
                    mse_pca.append(np.mean((pca_params[:, beta_idx] - betas_true[beta_idx])**2))
                else:
                    mse_pca.append(np.nan)

                if beta_idx < pls_params.shape[1]:
                    mse_pls.append(np.mean((pls_params[:, beta_idx] - betas_true[beta_idx])**2))
                else:
                    mse_pls.append(np.nan)


            key = (scenario_choice, tuple(betas_true))

            if key == ("baseline", (1,1,1,1,1)):
                if sample_size == 50:
                    color_pca = "#F08A9B"
                    color_pls = "#66BFBF"
                else:
                    color_pca = "#DC143C"
                    color_pls = "#008B8B"

            elif key == ("baseline", (5,0.5,0.5,0.5,0.5)):
                if sample_size == 50:
                    color_pca = "#6FBF6F"
                    color_pls = "#FF66FF"
                else:
                    color_pca = "#228B22"
                    color_pls = "#FF00FF"

            elif key == ("baseline", (3,-3,3,-3,3)):
                if sample_size == 50:
                    color_pca = "#FFD966"
                    color_pls = "#8FBCE6"
                else:
                    color_pca = "#FFD700"
                    color_pls = "#6495ED"

            elif key == ("baseline", (1,2,3,4,5)):
                if sample_size == 50:
                    color_pca = "#FF85C2"
                    color_pls = "#FFB347"
                else:
                    color_pca = "#FF1493"
                    color_pls = "#FFA500"

            elif key == ("extra_vars", (1,2,3,4,5)):
                if sample_size == 50:
                    color_pca = "#C8A2C8"
                    color_pls = "seagreen"
                else:
                    color_pca = "#800080"
                    color_pls = "darkgreen"

            elif key == ("omitted_vars", (1,2,3,4,5)):
                if sample_size == 50:
                    color_pca = "#C49A7A"
                    color_pls = "slateblue"
                else:
                    color_pca = "#8B4513"
                    color_pls = "darkblue"

            else:
                color_pca = "gray"
                color_pls = "gray"

            ax.plot(
                ratio_list,
                mse_pca,
                marker='o',
                linestyle='-',
                color=color_pca,
                label=f"PCR, n={sample_size}"
            )

            ax.plot(
                ratio_list,
                mse_pls,
                marker='o',
                linestyle='-',
                color=color_pls,
                label=f"PLSR, n={sample_size}"
            )


        for sample_size in sorted(set(w["sample_size"] for w in wyniki)):

            wybrane_ols = [
                w for w in wyniki
                if w["rho"] == rho
                and w["scenario"] == scenario_choice
                and w["sample_size"] == sample_size
            ]

            if not wybrane_ols:
                continue

            ols_vals = []

            for w in wybrane_ols:
                params = np.array(w["params_x_ols"])
                betas_true = np.array(w["betas_true"])

                if beta_idx < params.shape[1]:
                    ols_vals.append(np.mean((params[:, beta_idx] - betas_true[beta_idx])**2))

            ols_mse = np.mean(ols_vals) if ols_vals else np.nan

            if sample_size == 50:
                color_mnk = "#999999"
            else:
                color_mnk = "#000000"

            ax.axhline(
                ols_mse,
                linestyle=':',
                linewidth=2,
                color=color_mnk,
                label=f"MNK, n={sample_size}"
            )


        ax.set_title(f"ρ = {rho}")
        ax.set_xlabel("Explained variance (PCA)")
        ax.set_ylabel("MSE")

        ax.set_xticks(ratio_list)
        ax.set_xticklabels([f"{x:.1f}" for x in ratio_list])
        ax.tick_params(labelbottom=True)


        if i == 0:
            handles, labels = ax.get_legend_handles_labels()

            order_labels = [
                "PCR, n=50",
                "PCR, n=500",
                "PLSR, n=50",
                "PLSR, n=500",
                "MNK, n=50",
                "MNK, n=500"
            ]

            ordered_handles = []
            ordered_labels = []

            for lab in order_labels:
                for h, l in zip(handles, labels):
                    if l == lab:
                        ordered_handles.append(h)
                        ordered_labels.append(l)

            ax.legend(ordered_handles, ordered_labels)

    plt.tight_layout()
    plt.show()

# wrapper_mse

def wrapper_mse(wyniki):

    scenario_dropdown = Dropdown(
        options=sorted(list(set(w["scenario"] for w in wyniki))),
        description="scenario"
    )

    beta_slider = IntSlider(min=0, max=4, step=1, value=0, description="beta_idx")

    betas_dropdown = Dropdown(description="betas")

    def to_tuple(b):
        return tuple(float(x) for x in b)

    def get_betas_for_scenario(scenario):
        betas = {
            to_tuple(w["betas_true"])
            for w in wyniki
            if w["scenario"] == scenario
        }
        return {str(b): b for b in sorted(betas)}

    def update(*args):
        scenario = scenario_dropdown.value
        options = get_betas_for_scenario(scenario)

        betas_dropdown.options = options

        if options:
            betas_dropdown.value = list(options.values())[0]

    scenario_dropdown.observe(update, names='value')
    update()

    def plot_for_selection(scenario_choice, beta_idx, betas_choice):

        if betas_choice is None:
            return

        filt = [
            w for w in wyniki
            if w["scenario"] == scenario_choice
            and to_tuple(w["betas_true"]) == to_tuple(betas_choice)
        ]

        plot_mse(filt, beta_idx, scenario_choice)

    interact(
        plot_for_selection,
        scenario_choice=scenario_dropdown,
        beta_idx=beta_slider,
        betas_choice=betas_dropdown
    )

# plot_sign_accuracy

def plot_sign_accuracy(wyniki, scenario_choice):

    fig, axes = plt.subplots(2, 2, figsize=(12, 5.4), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, rho in enumerate(sorted(set(w["rho"] for w in wyniki))):

        ax = axes[i]

        for sample_size in sorted(set(w["sample_size"] for w in wyniki)):

            wybrane = [
                w for w in wyniki
                if w["sample_size"] == sample_size
                and w["rho"] == rho
                and w["scenario"] == scenario_choice
            ]

            if not wybrane:
                continue

            wybrane = sorted(wybrane, key=lambda w: w["components_ratio"])
            ratio_list = [w["components_ratio"] for w in wybrane]

            sign_pca = [w["sign_correct_pca"] for w in wybrane]
            sign_pls = [w["sign_correct_pls"] for w in wybrane]

            betas_true = np.array(wybrane[0]["betas_true"])
            key = (scenario_choice, tuple(betas_true))

            if key == ("baseline", (1,1,1,1,1)):
                if sample_size == 50:
                    color_pca, color_pls = "#F08A9B", "#66BFBF"
                else:
                    color_pca, color_pls = "#DC143C", "#008B8B"

            elif key == ("baseline", (5,0.5,0.5,0.5,0.5)):
                if sample_size == 50:
                    color_pca, color_pls = "#6FBF6F", "#FF66FF"
                else:
                    color_pca, color_pls = "#228B22", "#FF00FF"

            elif key == ("baseline", (3,-3,3,-3,3)):
                if sample_size == 50:
                    color_pca, color_pls = "#FFD966", "#8FBCE6"
                else:
                    color_pca, color_pls = "#FFD700", "#6495ED"

            elif key == ("baseline", (1,2,3,4,5)):
                if sample_size == 50:
                    color_pca, color_pls = "#FF85C2", "#FFB347"
                else:
                    color_pca, color_pls = "#FF1493", "#FFA500"

            elif key == ("extra_vars", (1,2,3,4,5)):
                if sample_size == 50:
                    color_pca, color_pls = "#C8A2C8", "seagreen"
                else:
                    color_pca, color_pls = "#800080", "darkgreen"

            elif key == ("omitted_vars", (1,2,3,4,5)):
                if sample_size == 50:
                    color_pca, color_pls = "#C49A7A", "slateblue"
                else:
                    color_pca, color_pls = "#8B4513", "darkblue"

            else:
                color_pca, color_pls = "gray", "gray"

            ax.plot(
                ratio_list,
                sign_pca,
                marker='o',
                linestyle='-',
                color=color_pca,
                label=f"PCR, n={sample_size}"
            )

            ax.plot(
                ratio_list,
                sign_pls,
                marker='o',
                linestyle='-',
                color=color_pls,
                label=f"PLSR, n={sample_size}"
            )


        for sample_size in sorted(set(w["sample_size"] for w in wyniki)):

            wybrane_ols = [
                w for w in wyniki
                if w["rho"] == rho
                and w["scenario"] == scenario_choice
                and w["sample_size"] == sample_size
            ]

            if not wybrane_ols:
                continue

            ols_acc = np.mean([w["sign_correct_ols"] for w in wybrane_ols])

            color_mnk = "#999999" if sample_size == 50 else "#000000"

            ax.axhline(
                ols_acc,
                linestyle=':',
                linewidth=2,
                color=color_mnk,
                label=f"MNK, n={sample_size}"
            )

        ax.set_title(f"ρ = {rho}")
        ax.set_xlabel("Explained variance (PCA)")
        ax.set_ylabel(r'$\alpha$')
        ax.set_xticks(ratio_list)
        ax.set_xticklabels([f"{x:.1f}" for x in ratio_list])
        ax.tick_params(labelbottom=True)

        if i == 0:
            handles, labels = ax.get_legend_handles_labels()

            order_labels = [
                "PCR, n=50",
                "PCR, n=500",
                "PLSR, n=50",
                "PLSR, n=500",
                "MNK, n=50",
                "MNK, n=500"
            ]

            ordered_handles = []
            ordered_labels = []

            for lab in order_labels:
                for h, l in zip(handles, labels):
                    if l == lab:
                        ordered_handles.append(h)
                        ordered_labels.append(l)

            ax.legend(ordered_handles, ordered_labels)

    plt.tight_layout()
    plt.show()

# wrapper_sign_accuracy

def wrapper_sign_accuracy(wyniki):

    scenario_dropdown = Dropdown(
        options=sorted(list(set(w["scenario"] for w in wyniki))),
        description="scenario"
    )

    betas_dropdown = Dropdown(description="betas")

    def to_tuple(b):
        return tuple(float(x) for x in b)

    def get_betas_for_scenario(scenario):
        betas = {
            to_tuple(w["betas_true"])
            for w in wyniki
            if w["scenario"] == scenario
        }
        return {str(b): b for b in sorted(betas)}

    def update(*args):
        scenario = scenario_dropdown.value
        options = get_betas_for_scenario(scenario)

        betas_dropdown.options = options

        if options:
            betas_dropdown.value = list(options.values())[0]

    scenario_dropdown.observe(update, names='value')
    update()

    def plot_for_selection(scenario_choice, betas_choice):

        if betas_choice is None:
            return

        filt = [
            w for w in wyniki
            if w["scenario"] == scenario_choice
            and to_tuple(w["betas_true"]) == to_tuple(betas_choice)
        ]

        plot_sign_accuracy(filt, scenario_choice)

    interact(
        plot_for_selection,
        scenario_choice=scenario_dropdown,
        betas_choice=betas_dropdown
    )

# plot_metrics_heatmap

def plot_metrics_heatmap(wyniki, betas_choice, scenario_choice, best_ratio):

    data = []

    for w in wyniki:
        if (
            w["components_ratio"] == best_ratio
            and tuple(w["betas_true"]) == betas_choice
            and w["scenario"] == scenario_choice
        ):
            data.append({
                "sample_size": w["sample_size"],
                "rho": w["rho"],
                "r2_ols": w["r2_ols"],
                "r2_pca": w["r2_pca"],
                "r2_pls": w["r2_pls"],
                "mse_cv_ols": w["mse_cv_ols"],
                "mse_cv_pca": w["mse_cv_pca"],
                "mse_cv_pls": w["mse_cv_pls"],
            })

    df = pd.DataFrame(data)

    if df.empty:
        print("Brak danych")
        return

    df_agg = df.groupby(["rho", "sample_size"]).mean().reset_index()

    fig, axes = plt.subplots(2, 3, figsize=(11, 5))

    metrics = [
        "r2_ols", "r2_pca", "r2_pls",
        "mse_cv_ols", "mse_cv_pca", "mse_cv_pls"
    ]

    titles = [
        r"$\overline{R^2}_{\mathrm{skorygowany}}$ - MNK",
        r"$\overline{R^2}_{\mathrm{skorygowany}}$ - PCR",
        r"$\overline{R^2}_{\mathrm{skorygowany}}$ - PLSR",
        r"$\overline{MSE}_{\mathrm{CV}}$ - MNK",
        r"$\overline{MSE}_{\mathrm{CV}}$ - PCR",
        r"$\overline{MSE}_{\mathrm{CV}}$ - PLSR"
    ]
    for ax, metric, title in zip(axes.flatten(), metrics, titles):

        heatmap_data = df_agg.pivot(
            index="rho",
            columns="sample_size",
            values=metric
        )

        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            ax=ax
        )

        ax.set_title(title)
        ax.set_xlabel("n")
        ax.set_ylabel("ρ")

    plt.tight_layout()
    plt.show()

# wrapper_heatmap

def wrapper_heatmap(wyniki):

    scenario_dropdown = Dropdown(
        options=sorted(list(set(w["scenario"] for w in wyniki))),
        description="scenario"
    )

    betas_dropdown = Dropdown(description="betas")

    def to_tuple(b):
        return tuple(float(x) for x in b)

    def get_betas_for_scenario(scenario):
        betas = {
            to_tuple(w["betas_true"])
            for w in wyniki
            if w["scenario"] == scenario
        }
        return {str(b): b for b in sorted(betas)}

    def update(*args):
        scenario = scenario_dropdown.value
        options = get_betas_for_scenario(scenario)

        betas_dropdown.options = options

        if options:
            betas_dropdown.value = list(options.values())[0]

    scenario_dropdown.observe(update, names='value')
    update()

    def plot_for_selection(scenario_choice, betas_choice):

        if betas_choice is None:
            return

        plot_metrics_heatmap(
            wyniki=wyniki,
            betas_choice=to_tuple(betas_choice),
            scenario_choice=scenario_choice,
            best_ratio = 0.6
        )

    interact(
        plot_for_selection,
        scenario_choice=scenario_dropdown,
        betas_choice=betas_dropdown
    )
