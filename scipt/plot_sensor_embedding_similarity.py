import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


DEFAULT_SENSOR_LABELS = [
    "S2", "S3", "S4", "S7", "S8", "S9", "S11", "S12",
    "S13", "S14", "S15", "S17", "S20", "S21",
]


def _find_latest_checkpoint(project_root, sub_dataset):
    patterns = [
        os.path.join(project_root, "logs", "**", f"best_model_{sub_dataset}_*.pkl"),
        os.path.join(project_root, "logs", "**", f"model_{sub_dataset}_*.pkl"),
        os.path.join(project_root, "trials", f"best_model_{sub_dataset}_*.pkl"),
        os.path.join(project_root, "trials", f"model_{sub_dataset}_*.pkl"),
        os.path.join(project_root, "trials", f"model_{sub_dataset}.pkl"),
    ]

    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern, recursive=True))

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found for {sub_dataset} under logs/ or trials/."
        )
    return max(candidates, key=os.path.getmtime)


def _extract_sensor_embedding_weight(model):
    if not hasattr(model, "spatial_gat") or model.spatial_gat is None:
        raise ValueError("Model has no spatial_gat, cannot read sensor embeddings.")

    spatial_gat = model.spatial_gat
    if not hasattr(spatial_gat, "sensor_embedding"):
        raise ValueError("spatial_gat has no sensor_embedding parameter.")

    weight = spatial_gat.sensor_embedding.weight.detach().cpu()
    if weight.ndim != 2:
        raise ValueError(
            f"Expected embedding weight shape (num_sensors, embed_dim), got {tuple(weight.shape)}"
        )
    return weight


def _build_cosine_similarity(embedding_weight):
    normed = F.normalize(embedding_weight, p=2, dim=-1)
    sim = torch.matmul(normed, normed.transpose(0, 1))
    return sim.numpy()


def _plot_similarity_heatmap(similarity, labels, out_path, title):
    plt.figure(figsize=(8.2, 6.6), dpi=150)
    ax = plt.gca()

    img = ax.imshow(similarity, cmap="Blues", vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Draw subtle grid lines to mimic paper-style matrix blocks.
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8, alpha=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title(title, fontsize=11)
    cbar = plt.colorbar(img, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _build_adjacency_matrix(model, similarity):
    spatial_gat = model.spatial_gat
    graph_mode = getattr(spatial_gat, "graph_mode", "dynamic_knn")
    num_sensors = similarity.shape[0]

    if graph_mode == "dynamic_knn":
        topk = int(getattr(spatial_gat, "topk", num_sensors))
        topk = max(1, min(topk, num_sensors))

        # Match model behavior: select top-k neighbors per sensor based on cosine similarity.
        topk_indices = np.argpartition(-similarity, kth=topk - 1, axis=1)[:, :topk]
        adjacency = np.zeros((num_sensors, num_sensors), dtype=np.int32)
        row_index = np.arange(num_sensors)[:, None]
        adjacency[row_index, topk_indices] = 1
        np.fill_diagonal(adjacency, 1)
        return adjacency

    if graph_mode == "path":
        adjacency = np.eye(num_sensors, dtype=np.int32)
        if num_sensors > 1:
            idx = np.arange(num_sensors - 1)
            adjacency[idx, idx + 1] = 1
            adjacency[idx + 1, idx] = 1
        return adjacency

    raise ValueError(f"Unsupported graph_mode for adjacency plotting: {graph_mode}")


def _plot_adjacency_matrix(adjacency, labels, out_path, title):
    plt.figure(figsize=(8.6, 6.9), dpi=150)
    ax = plt.gca()

    ax.imshow(adjacency, cmap="Blues", vmin=0, vmax=1, alpha=0.22)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=14, fontweight="bold")
    ax.set_yticklabels(labels, fontsize=14, fontweight="bold")
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)

    for i in range(adjacency.shape[0]):
        for j in range(adjacency.shape[1]):
            ax.text(j, i, str(int(adjacency[i, j])), ha="center", va="center", fontsize=16, color="#222222")

    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="#d0d0d0", linestyle="-", linewidth=0.7, alpha=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title(title, fontsize=12, pad=24, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _plot_connection_graph(adjacency, labels, out_path, title):
    num_nodes = adjacency.shape[0]
    angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
    radius = 1.0
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)

    plt.figure(figsize=(7.4, 7.4), dpi=150)
    ax = plt.gca()
    node_fill_color = "#c6ddab"
    node_edge_color = "#5f7f4a"
    node_text_color = "#2f4a26"

    # Draw edges (ignore self-loops in topology plot).
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            has_ij = adjacency[i, j] == 1
            has_ji = adjacency[j, i] == 1
            if not (has_ij or has_ji):
                continue
            alpha = 0.85 if (has_ij and has_ji) else 0.5
            lw = 1.8 if (has_ij and has_ji) else 1.2
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color="#6c7ea0", linewidth=lw, alpha=alpha, zorder=1)

    # Draw nodes and labels.
    ax.scatter(xs, ys, s=1400, c=node_fill_color, edgecolors=node_edge_color, linewidths=2.6, zorder=2)
    for idx, (x, y) in enumerate(zip(xs, ys)):
        ax.text(x, y, labels[idx], ha="center", va="center", color=node_text_color, fontsize=15, fontweight="bold", zorder=3)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_aspect("equal")
    ax.axis("off")
    margin = 1.25
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot 14-sensor embedding cosine similarity heatmap")
    parser.add_argument("--model-path", type=str, default="", help="Path to checkpoint .pkl")
    parser.add_argument("--sub-dataset", type=str, default="FD001", help="FD001/FD002/FD003/FD004")
    parser.add_argument("--out-path", type=str, default="", help="Output image path (png/svg)")
    parser.add_argument("--title", type=str, default="Cosine similarity between embeddings")
    parser.add_argument("--save-adjacency-svg", dest="save_adjacency_svg", action="store_true", default=True)
    parser.add_argument("--no-save-adjacency-svg", dest="save_adjacency_svg", action="store_false")
    parser.add_argument("--adj-out-path", type=str, default="", help="Output path for adjacency matrix svg")
    parser.add_argument("--adj-title", type=str, default="Adjacency matrix A")
    parser.add_argument("--save-connection-svg", dest="save_connection_svg", action="store_true", default=True)
    parser.add_argument("--no-save-connection-svg", dest="save_connection_svg", action="store_false")
    parser.add_argument("--conn-out-path", type=str, default="", help="Output path for connection graph svg")
    parser.add_argument("--conn-title", type=str, default="Sensor connection graph")
    parser.add_argument("--no-cuda", action="store_true", default=False)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    model_path = args.model_path.strip() if args.model_path else ""
    if not model_path:
        model_path = _find_latest_checkpoint(PROJECT_ROOT, args.sub_dataset)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    model = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()

    embedding_weight = _extract_sensor_embedding_weight(model)
    similarity = _build_cosine_similarity(embedding_weight)

    labels = DEFAULT_SENSOR_LABELS
    if similarity.shape[0] != len(labels):
        labels = [f"S{i+1}" for i in range(similarity.shape[0])]

    if args.out_path:
        out_path = args.out_path
    else:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        out_path = os.path.join(
            PROJECT_ROOT,
            "figure",
            f"{args.sub_dataset}_{model_name}_embedding_similarity.svg",
        )

    _plot_similarity_heatmap(
        similarity=similarity,
        labels=labels,
        out_path=out_path,
        title=args.title,
    )

    adjacency_out_path = ""
    connection_out_path = ""
    adjacency = None
    if args.save_adjacency_svg or args.save_connection_svg:
        adjacency = _build_adjacency_matrix(model, similarity)

    if args.save_adjacency_svg:
        if args.adj_out_path:
            adjacency_out_path = args.adj_out_path
        else:
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            adjacency_out_path = os.path.join(
                PROJECT_ROOT,
                "figure",
                f"{args.sub_dataset}_{model_name}_adjacency_matrix.svg",
            )
        _plot_adjacency_matrix(
            adjacency=adjacency,
            labels=labels,
            out_path=adjacency_out_path,
            title=args.adj_title,
        )

    if args.save_connection_svg:
        if args.conn_out_path:
            connection_out_path = args.conn_out_path
        else:
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            connection_out_path = os.path.join(
                PROJECT_ROOT,
                "figure",
                f"{args.sub_dataset}_{model_name}_connection_graph.svg",
            )
        _plot_connection_graph(
            adjacency=adjacency,
            labels=labels,
            out_path=connection_out_path,
            title=args.conn_title,
        )

    print(f"model_path:{model_path}")
    print(f"output_image:{out_path}")
    if adjacency_out_path:
        print(f"adjacency_svg:{adjacency_out_path}")
    if connection_out_path:
        print(f"connection_svg:{connection_out_path}")
    print(f"matrix_shape:{similarity.shape}")


if __name__ == "__main__":
    main()
