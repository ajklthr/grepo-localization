import os
from hashlib import sha1
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from grepo_localization.corpus import API, REVISION, GrepoCorpus, IssueRef, release_manifest
from grepo_localization.graphs import graph_for_issue
from grepo_localization.inspection import to_networkx_issue_graph

NODE_TYPES = {0: "Directory", 1: "File", 2: "Python file", 3: "Class", 4: "Function"}
COLORS = {"Anchor": "#277C89", "Expanded candidate": "#5A69AF",
          "Changed by the fix": "#C75E2B", "Context": "#889894"}


@st.cache_resource(show_spinner="Loading the GREPO repository list…")
def open_corpus(cache_root, revision):
    return GrepoCorpus(release_manifest(API, revision), Path(cache_root), API, revision,
                       trust_graph_files=True)


def select_repository(corpus):
    with st.sidebar:
        st.header("Repositories")
        names = corpus.repositories()
        if not names:
            st.info("No repositories are available.")
            return None
        repo = st.selectbox("Repository", names, index=names.index("csvkit") if "csvkit" in names else 0)
        ready, size = corpus.repository_status(repo)
        st.caption(f"{'Downloaded' if ready else 'Available to download'} · {size / 1e6:,.1f} MB")
        if not ready and st.button("Download repository", type="primary"):
            try:
                with st.spinner(f"Downloading and verifying {repo}…"):
                    corpus.download_repository(repo)
            except (OSError, ValueError) as exc:
                st.error(f"Download failed. You can retry. {exc}")
            else:
                st.rerun()
    if not ready:
        st.info(f"Download {repo} to browse its issues.")
        return None
    return repo


def plot_issue(graph, anchors, gold, selected=None, max_nodes=100):
    priority = sorted(graph, key=lambda node: (node not in gold, node not in anchors, node))
    visible_nodes = priority[:max_nodes]
    if selected in graph and selected not in visible_nodes:
        visible_nodes[-1] = selected
    view = nx.Graph(graph.subgraph(visible_nodes))
    layout = nx.spring_layout(view, seed=42)
    x_edges, y_edges = [], []
    for a, b in view.edges:
        x_edges.extend((layout[a][0], layout[b][0], None))
        y_edges.extend((layout[a][1], layout[b][1], None))
    fig = go.Figure(go.Scatter(x=x_edges, y=y_edges, mode="lines", hoverinfo="skip",
                              line={"color": "#C7D1CE", "width": 1}, showlegend=False))
    groups = {label: [] for label in COLORS}
    for node in view:
        if node in gold:
            label = "Changed by the fix"
        elif node in anchors:
            label = "Anchor"
        elif graph.nodes[node]["candidate"]:
            label = "Expanded candidate"
        else:
            label = "Context"
        groups[label].append(node)
    for label, nodes in groups.items():
        if nodes:
            fig.add_scatter(x=[layout[node][0] for node in nodes],
                            y=[layout[node][1] for node in nodes], mode="markers",
                            name=label, customdata=[[node] for node in nodes],
                            hovertemplate="Node %{customdata[0]}<extra>%{fullData.name}</extra>",
                            marker={"size": 13, "color": COLORS[label],
                                    "line": {"color": "#FFFFFF", "width": 1}})
    if selected in view:
        fig.add_scatter(
            x=[layout[selected][0]],
            y=[layout[selected][1]],
            mode="markers",
            name="Selected node",
            customdata=[[selected]],
            hovertemplate="Selected node %{customdata[0]}<extra></extra>",
            marker={"size": 24, "symbol": "circle-open", "color": "#F2A900",
                    "line": {"color": "#F2A900", "width": 4}},
        )
    fig.update_layout(height=445, dragmode="pan", clickmode="event+select", margin=dict(l=8, r=8, t=8, b=0),
                      legend=dict(orientation="h", y=-0.04), hovermode="closest")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")
    return fig, view.number_of_nodes()


def node_table(node_ids, types, anchors, gold, candidates):
    return pd.DataFrame([
        {"Node ID": node, "Type": NODE_TYPES.get(int(types[node]), "Other"),
         "Anchor": node in anchors, "Candidate": node in candidates, "Changed": node in gold}
        for node in sorted(node_ids)
    ], columns=["Node ID", "Type", "Anchor", "Candidate", "Changed"])


def filter_table_values(table, query):
    query = query.strip()
    if not query:
        return table
    values = table.astype("string")
    matches = values.apply(
        lambda column: column.str.contains(query, case=False, regex=False, na=False)
    ).any(axis=1)
    return table.loc[matches]


def filter_table_columns(table, selections):
    filtered = table
    for column, selected_values in selections.items():
        if selected_values:
            filtered = filtered[filtered[column].isin(selected_values)]
    return filtered


def show_report(corpus, ref, record):
    with st.expander("Issue report and search queries", expanded=True):
        try:
            report_record = corpus.issue(ref, include_report=True)
            for report in report_record.reports:
                st.subheader(f"#{report.get('number', '?')} · {report.get('title', '')}")
                st.text(report.get("body") or "No issue description is available.")
            if not report_record.reports and report_record.fix_pr:
                pr = report_record.fix_pr
                st.caption(f"No linked issue text. Fix PR #{pr['number']} · {pr['title']}")
                st.text(pr.get("body") or "")
        except (OSError, ValueError, IndexError, KeyError) as exc:
            st.warning(f"Issue report unavailable. {exc}")
        if record.search_queries:
            st.markdown("**GREPO search queries**")
            for number, query in enumerate(record.search_queries, start=1):
                with st.container(border=True):
                    st.caption(f"Query {number}")
                    st.markdown(query)


def select_plot_node(chart_key, node_key):
    points = st.session_state[chart_key]["selection"]["points"]
    if points and points[-1].get("customdata"):
        st.session_state[node_key] = int(points[-1]["customdata"][0])


def select_table_node(table_key, node_key, node_ids):
    rows = st.session_state[table_key]["selection"]["rows"]
    if rows:
        st.session_state[node_key] = int(node_ids[rows[0]])


def select_node_id(action_key, node_key, node_ids):
    click = st.session_state.get(action_key)
    if click is not None:
        st.session_state[node_key] = int(node_ids[click["row"]])


def show_nodes(corpus, ref, graph, anchors, gold, has_gold):
    candidates = {node for node, attrs in graph.nodes(data=True) if attrs["candidate"]}
    types = corpus.repository_graph(ref.repo).type.flatten()
    in_bounds = lambda nodes: {node for node in nodes if 0 <= node < types.numel()}
    inspectable = set(graph) | anchors | gold
    key = f"{ref.repo}:{ref.issue_id}"
    node_key = f"{key}:node"
    graph_column, details_column = st.columns([1.18, 1], gap="large")
    with graph_column:
        st.subheader("Issue graph")
        selected_node = st.session_state.get(node_key)
        if graph:
            fig, shown = plot_issue(graph, anchors, gold, selected_node)
            chart_key = f"{key}:graph"
            st.plotly_chart(fig, key=chart_key,
                            on_select=lambda: select_plot_node(chart_key, node_key),
                            selection_mode="points")
            st.caption(f"{shown} of {graph.number_of_nodes()} graph nodes shown. Lines combine directions and edge types.")
            if selected_node is not None and selected_node not in graph:
                st.caption(f"Selected node {selected_node} is outside this issue graph.")
        else:
            st.info("No active anchors produced an issue graph.")
        scope = st.radio("Node list", ["Candidates", "Anchors", "Changed by the fix", "All graph nodes"],
                         horizontal=True, key=f"{key}:scope")
        visible = {"Candidates": candidates, "Anchors": anchors,
                   "Changed by the fix": gold, "All graph nodes": set(graph)}[scope]
        table = node_table(in_bounds(visible), types, anchors, gold, candidates)
        if scope == "Changed by the fix":
            table["Retrieval"] = table["Candidate"].map({True: "In candidates", False: "Missing from candidates"})
            if not has_gold:
                st.caption("This issue has no released gold labels.")
        if not table.empty:
            filter_column, page_size_column = st.columns([3, 1], vertical_alignment="bottom")
            with filter_column:
                query = st.text_input(
                    "Filter table values",
                    type="search",
                    placeholder="Search node IDs, types, or statuses",
                    key=f"{key}:filter:{scope}",
                )
            with page_size_column:
                rows_per_page = st.selectbox(
                    "Rows per page", (25, 50, 100), index=1,
                    key=f"{key}:page-size:{scope}",
                )
            column_filters = {}
            with st.expander("Filter by column"):
                st.caption("Choose values within a column. Filters across columns are combined.")
                filter_columns = st.columns(2)
                for index, column in enumerate(table.columns):
                    options = sorted(table[column].drop_duplicates().tolist(), key=lambda value: str(value).casefold())
                    with filter_columns[index % 2]:
                        column_filters[column] = st.multiselect(
                            column,
                            options,
                            key=f"{key}:column-filter:{scope}:{column}",
                        )
            filtered_table = filter_table_columns(table, column_filters)
            filtered_table = filter_table_values(filtered_table, query)
            if filtered_table.empty:
                st.info("No node values match this filter.")
            else:
                filter_token = sha1(
                    repr((query.casefold(), column_filters)).encode()
                ).hexdigest()[:10]
                table_slot = st.empty()
                total_pages = max(1, (len(filtered_table) + rows_per_page - 1) // rows_per_page)
                with st.container(horizontal_alignment="right"):
                    page = st.pagination(
                        total_pages,
                        key=f"{key}:page:{scope}:{rows_per_page}:{filter_token}",
                    )
                start = (page - 1) * rows_per_page
                page_table = filtered_table.iloc[start:start + rows_per_page]
                table_key = f"{key}:table:{scope}:{rows_per_page}:{page}:{filter_token}"
                action_key = f"{table_key}:node-id"
                node_ids = page_table["Node ID"].tolist()
                display_table = page_table.copy()
                display_table["Node ID"] = display_table["Node ID"].astype(str)
                table_slot.dataframe(
                    display_table,
                    height=360,
                    hide_index=True,
                    key=table_key,
                    on_select=lambda: select_table_node(
                        table_key, node_key, node_ids
                    ),
                    selection_mode="single-row",
                    column_config={
                        "Node ID": st.column_config.ButtonColumn(
                            "Node ID",
                            help="Select this node in the graph and code viewer.",
                            pinned=True,
                            type="tertiary",
                            on_click=select_node_id,
                            args=(action_key, node_key, node_ids),
                            key=action_key,
                        ),
                    },
                    alt=f"{scope} node list",
                )
                st.caption(
                    f"Showing {start + 1}–{min(start + rows_per_page, len(filtered_table))} "
                    f"of {len(filtered_table)} matching nodes."
                )
        else:
            st.caption("No nodes in this view.")

    with details_column:
        st.subheader("Code viewer")
        st.selectbox("Inspect node ID", sorted(inspectable), index=None,
                     placeholder="Pick a node from the graph or table", key=node_key)
        node_id = st.session_state.get(node_key)
        if node_id is None:
            st.info("Select a node to see its historical source.")
            return
        status = []
        if node_id in anchors:
            status.append("Starting anchor")
        if node_id in gold:
            status.append("Changed by the fix")
        status.append("In candidates" if node_id in candidates else "Outside candidates")
        st.caption(" · ".join(status))
        try:
            with st.spinner("Loading historical source. The first lookup downloads the source table…"):
                node = corpus.nodes(ref.repo, [node_id])[node_id]
        except (OSError, ValueError, KeyError) as exc:
            st.error(f"Source unavailable. {exc}")
            st.button("Retry source lookup", key=f"{key}:retry")
            return
        st.markdown(f"**{node.name or 'Unnamed node'}**")
        st.text(node.path)
        if node.text:
            st.code(node.text, language="python")
        else:
            st.caption("GREPO has no stored source text for this node.")


def main():
    st.set_page_config(page_title="GREPO Issue Browser", page_icon="🧩", layout="wide")
    st.title("GREPO Issue Browser")
    try:
        corpus = open_corpus(os.environ.get("GREPO_CACHE_DIR", "grepo_workspace"), REVISION)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Could not load the GREPO release. {exc}")
        st.button("Retry repository list")
        return
    repo = select_repository(corpus)
    if repo is None:
        return
    try:
        refs = corpus.issue_refs(repo)
        if not refs:
            st.info("This repository has no released issues.")
            return
        issue_ids = sorted(ref.issue_id for ref in refs)
        default = issue_ids.index(116) if repo == "csvkit" and 116 in issue_ids else 0
        with st.sidebar:
            issue_id = st.selectbox("GREPO issue ID", issue_ids, index=default, key=f"issues:{repo}")
        ref = IssueRef(repo, issue_id)
        record = corpus.issue(ref)
        data = graph_for_issue(corpus, ref)
        gold = corpus.gold(ref)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Could not open the issue. {exc}")
        st.button("Retry issue")
        return
    st.header(f"{repo} / {issue_id}")
    candidate_ids = set(data.original_node_id[data.candidate_mask].tolist())
    a, b, c = st.columns(3)
    a.metric("Starting anchors", len(record.extractor_anchor_ids))
    b.metric("Candidates", len(candidate_ids))
    c.metric("Changed nodes in candidates", f"{len(candidate_ids & (gold or set()))} / {len(gold)}" if gold else "—")
    show_report(corpus, ref, record)
    show_nodes(corpus, ref, to_networkx_issue_graph(data), set(record.extractor_anchor_ids),
               set(gold or ()), gold is not None)


if __name__ == "__main__":
    main()
