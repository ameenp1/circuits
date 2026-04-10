from circuits.analysis.steer import (
    batchify,
    format_token,
    prepare_circuits_for_interchange_interventions,
    run_zero_intervention,
)
from circuits.tracing.trace import convert_inputs_to_circuits
from transformers import AutoTokenizer
from util.subject import Subject, llama31_8B_instruct_config

subject = Subject(llama31_8B_instruct_config)
tokenizer = subject.tokenizer

# example input, always just do one example
prompts = [f"Just give me the answer. What is the capital of the state containing Dallas?"]
seed_responses = ["Answer:"] * len(prompts)
labels = ["Sacramento"]

# convert to dataframes
from circuits.tracing.clja import ADAGConfig

config = ADAGConfig(
    verbose=False,
    topk_neurons=200,
    use_relp_grad=False,
)
circuit_data = convert_inputs_to_circuits(
    subject,
    tokenizer,
    prompts,
    config=config,
    seed_responses=seed_responses,
    labels=labels,
    num_datapoints=1,  # always 1
    batch_size=1,  # always 1
    k=5,  # user can set this, # top-k logits
    ignore_bos=True,
    use_rollout=False,
)
df_node = circuit_data.df_node
df_edge = circuit_data.df_edge
cis = circuit_data.cis
attention_masks = circuit_data.attention_masks

# do clustering
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
data_circuit, df_node_embedded, df_edge_embedded = await build_circuit_visualization(
    df_node,
    df_edge,
    tokenizer,
    mode="attr + contrib",  # user can pick from attr, contrib, attr + contrib, attr x contrib, random
    do_label=False,  # user can set this
    do_layernorm=True,  # fixed
    n_clusters=16,  # user can set this
    do_average_over_examples=False,
    drop_input_tokens=True,
    sum_over_tokens=False,
    get_desc=True,
    give_top_example=False,
    include_attr_contrib=True,
    return_clustered_dfs=True,
    do_one_cluster_per_neuron=False,
)

# make steering-ready df
clustered_df = df_node.copy()
node_to_cluster = df_node_embedded.set_index("input_variable")["cluster"].to_dict()
clustered_df["cluster"] = clustered_df.apply(
    lambda row: node_to_cluster.get(f"{row.layer},{row.token},{row.neuron}", None), axis=1
)
cluster_to_attribution_sum = clustered_df.groupby("cluster").attribution.sum().to_dict()

# USER DEFINE SOME SET OF INPUT_VARIABLES TO RUN STEERING ON
user_input_variables: list[str] = ["23,26,8079", "21,26,4924", "21,26,3093"]
# e.g., user can modify this by selecting nodes from the table in the ui.
# # it should start empty. and it should run steering and redisplay the table when the user changes it.


def do_steering(user_input_variables: list[str]):
    # get subset df
    subset_df = clustered_df[
        (clustered_df.input_variable.isin(user_input_variables))
        & (clustered_df.layer != -1)
        & (clustered_df.layer != 32)
    ]

    # do steering
    pairs = prepare_circuits_for_interchange_interventions(
        subject=subject,
        cis=cis,
        attention_masks=attention_masks,
        labels=labels,
        df_node=subset_df,
        needed_pairs=1,
        allow_same_label=True,
        ignore_source=True,
    )
    batch_size = 1
    batches = batchify(pairs, batch_size)
    data = run_zero_intervention(
        subject, batches, batch_size, multiplier=0.0
    )  # user can control multiplier, it's 0.0 by default

    # show user this
    intervened_probs = data[0]["intervened_probs"]
    original_probs = data[0]["original_probs"]
    diff_probs = intervened_probs - original_probs

    top_100_diffs = torch.topk(diff_probs, k=100)
    top_100_diff_tokens = [
        format_token(token, subject.tokenizer) for token in top_100_diff_probs.indices
    ]
    top_100_diff_probs = top_100_diff_probs.values

    top_100_original = torch.topk(original_probs, k=100)
    top_100_original_tokens = [
        format_token(token, subject.tokenizer) for token in top_100_original.indices
    ]
    top_100_original_probs = top_100_original.values

    top_100_intervened = torch.topk(intervened_probs, k=100)
    top_100_intervened_tokens = [
        format_token(token, subject.tokenizer) for token in top_100_intervened.indices
    ]
    top_100_intervened_probs = top_100_intervened.values

    # show user this in a nice scrollable table that doesn't take up all the screen...
    return (
        top_100_diff_tokens,
        top_100_diff_probs,
        top_100_original_tokens,
        top_100_original_probs,
        top_100_intervened_tokens,
        top_100_intervened_probs,
    )
