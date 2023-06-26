import xml.etree.ElementTree as ET
import argparse
import awswrangler as wr
import pandas as pd
import random
import torch

from tqdm import tqdm
from grants_tagger_light.predict import predict_tags as predict_tags_bertmesh
from grants_tagger_light.models.xlinear import MeshXLinear

random.seed(42)
torch.manual_seed(42)


def load_mesh_terms_from_file(mesh_terms_list_path: str):
    with open(mesh_terms_list_path, "r") as f:
        mesh_terms = f.readlines()
    mesh_terms = [term.strip() for term in mesh_terms]
    return mesh_terms


def load_mesh_tree_node_letters(mesh_codes_list_path: str):
    """
    Example file:
    Information Sources: L
    Phenomena and Processes: G
    Geographicals: Z
    Diseases: C

    This functions extracts the letters from the file. The letters
    represent a top level descriptor from the MeSH tree ontology.
    Below is an example MeSH term for the letter "C", i.e. diseases:

    Tree num : C10.597.606.762.175
    Code : D066190
    Name : Allesthesia
    """
    with open(mesh_codes_list_path, "r") as f:
        mesh_codes = f.readlines()
    mesh_codes = [code.strip().split(": ")[1] for code in mesh_codes]
    return mesh_codes


def _extract_data(mesh_elem):
    # TreeNumberList e.g. A11.118.637.555.567.550.500.100
    tree_number = mesh_elem[-2][0].text
    # DescriptorUI e.g. M000616943
    code = mesh_elem[0].text
    # DescriptorName e.g. Mucosal-Associated Invariant T Cells
    name = mesh_elem[1][0].text

    return tree_number, code, name


def find_subnames_from_terms(mesh_metadata_path: str, mesh_terms_list_path: str):
    """
    Given a path to a file containing a list of MeSH terms and a path to a file
    containing MeSH metadata, returns a list of all MeSH terms that are subnames
    of the MeSH terms in the input file.

    Args:
        mesh_metadata_path (str): The path to the MeSH metadata file.
        mesh_terms_list_path (str): The file containing the list of MeSH terms
                                    to filter by.

    Returns:
        List[str]: A list of all MeSH terms that are subnames
                   of the MeSH terms in the input file.
    """
    mesh_tree = ET.parse(mesh_metadata_path)
    mesh_terms = load_mesh_terms_from_file(mesh_terms_list_path)
    root = mesh_tree.getroot()

    # Do 1st pass to get all their codes
    top_level_tree_numbers = []
    pbar = tqdm(root)
    pbar.set_description("Finding top level tree numbers from terms")

    found_terms = []

    for mesh_elem in pbar:
        try:
            tree_number, code, name = _extract_data(mesh_elem)
        except IndexError:
            continue

        if name not in mesh_terms:
            continue
        else:
            top_level_tree_numbers.append(tree_number)
            found_terms.append(name)

    assert len(found_terms) == len(
        mesh_terms
    ), "Not all terms were found in MeSH tree. Terms that were not found : {}".format(
        set(mesh_terms) - set(found_terms)
    )

    # Do 2nd pass to collect all names that are in the same tree as the ones we found
    all_subnames = []
    pbar = tqdm(root)
    pbar.set_description("Finding subnames")
    for mesh_elem in pbar:
        try:
            curr_tree_number, _, name = _extract_data(mesh_elem)
        except IndexError:
            continue

        for top_level_tree_number in top_level_tree_numbers:
            if curr_tree_number.startswith(top_level_tree_number):
                all_subnames.append(name)
                break

    return all_subnames


def find_subnames_from_tree_nodes(mesh_metadata_path: str, mesh_tree_nodes_path: str):
    mesh_tree = ET.parse(mesh_metadata_path)
    tree_node_letters = load_mesh_tree_node_letters(mesh_tree_nodes_path)
    root = mesh_tree.getroot()

    # Do 1st pass to get all their codes
    top_level_tree_numbers = []
    pbar = tqdm(root)
    pbar.set_description("Finding top level tree numbers from tree nodes")

    found_terms = []
    for mesh_elem in pbar:
        try:
            tree_number, _, name = _extract_data(mesh_elem)
        except IndexError:
            continue

        for tree_node_letter in tree_node_letters:
            if tree_number.startswith(tree_node_letter):
                top_level_tree_numbers.append(tree_number)
                found_terms.append(name)
                break

    # Do 2nd pass to collect all names that are in the same tree as the ones we found
    all_subnames = []
    pbar = tqdm(root)
    pbar.set_description("Finding subnames")
    for mesh_elem in pbar:
        try:
            curr_tree_number, _, name = _extract_data(mesh_elem)
        except IndexError:
            continue

        for top_level_tree_number in top_level_tree_numbers:
            if curr_tree_number.startswith(top_level_tree_number):
                all_subnames.append(name)
                break

    return all_subnames


def create_comparison_csv(
    s3_url: str,
    num_parquet_files_to_consider: int,
    num_samples_per_cat: int,
    mesh_metadata_path: str,
    mesh_terms_list_path: str,
    mesh_tree_letters_list_path: str,
    pre_annotate_bertmesh: bool,
    bertmesh_path: str,
    bertmesh_thresh: float,
    pre_annotate_xlinear: bool,
    xlinear_path: str,
    xlinear_label_binarizer_path: str,
    xlinear_thresh: float,
    output_path: str,
):
    subnames_from_terms = find_subnames_from_terms(
        mesh_metadata_path, mesh_terms_list_path
    )
    subnames_from_tree_letters = find_subnames_from_tree_nodes(
        mesh_metadata_path, mesh_tree_letters_list_path
    )

    subnames = set(subnames_from_terms).union(set(subnames_from_tree_letters))

    parquet_files = wr.s3.list_objects(s3_url)
    random.shuffle(parquet_files)

    grants = []

    for idx in tqdm(range(num_parquet_files_to_consider)):
        grants.append(
            wr.s3.read_parquet(
                parquet_files[idx],
            )
        )

    all_grants = pd.concat(grants)

    # Filter out rows where abstract is na
    all_grants = all_grants[~all_grants["abstract"].isna()]

    # Do stratified sampling based on for_first_level_name column
    grants_sample = all_grants.groupby("for_first_level_name", group_keys=False).apply(
        lambda x: x.sample(min(len(x), num_samples_per_cat))
    )

    abstracts = grants_sample["abstract"].tolist()

    # Annotate with bertmesh
    if pre_annotate_bertmesh:
        tags = predict_tags_bertmesh(
            abstracts,
            bertmesh_path,
            return_labels=True,
            threshold=bertmesh_thresh,
        )

        grants_sample["bertmesh_terms"] = tags
        # Keep 1st elem of each list (above func returns them nested)
        grants_sample["bertmesh_terms"] = grants_sample["bertmesh_terms"].apply(
            lambda x: x[0]
        )

    # Annotate with xlinear
    if pre_annotate_xlinear:
        model = MeshXLinear(
            model_path=xlinear_path,
            label_binarizer_path=xlinear_label_binarizer_path,
        )

        tags = model(X=abstracts, threshold=xlinear_thresh)

        grants_sample["xlinear_terms"] = tags

    if pre_annotate_bertmesh:
        # Filter out rows where none of the bertmesh tags are in the subnames list
        grants_sample = grants_sample[
            grants_sample["bertmesh_terms"].apply(
                lambda x: any([tag in subnames for tag in x])
            )
        ]

    # Output df to csv
    grants_sample.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-url", type=str)
    parser.add_argument("--num-parquet-files-to-consider", type=int, default=1)
    parser.add_argument("--num-samples-per-cat", type=int, default=10)
    parser.add_argument("--mesh-metadata-path", type=str)
    parser.add_argument("--mesh-terms-list-path", type=str)
    parser.add_argument("--mesh-tree-letters-list-path", type=str)
    parser.add_argument("--pre-annotate-bertmesh", action="store_true")
    parser.add_argument(
        "--bertmesh-path", type=str, default="Wellcome/WellcomeBertMesh"
    )
    parser.add_argument("--bertmesh-thresh", type=float, default=0.5)
    parser.add_argument("--pre-annotate-xlinear", action="store_true")
    parser.add_argument("--xlinear-path", type=str)
    parser.add_argument("--xlinear-label-binarizer-path", type=str)
    parser.add_argument("--xlinear-thresh", type=float, default=0.5)
    parser.add_argument("--output-path", type=str)

    args = parser.parse_args()

    create_comparison_csv(
        s3_url=args.s3_url,
        num_parquet_files_to_consider=args.num_parquet_files_to_consider,
        num_samples_per_cat=args.num_samples_per_cat,
        mesh_metadata_path=args.mesh_metadata_path,
        mesh_terms_list_path=args.mesh_terms_list_path,
        mesh_tree_letters_list_path=args.mesh_tree_letters_list_path,
        pre_annotate_bertmesh=args.pre_annotate_bertmesh,
        bertmesh_path=args.bertmesh_path,
        bertmesh_thresh=args.bertmesh_thresh,
        pre_annotate_xlinear=args.pre_annotate_xlinear,
        xlinear_path=args.xlinear_path,
        xlinear_label_binarizer_path=args.xlinear_label_binarizer_path,
        xlinear_thresh=args.xlinear_thresh,
        output_path=args.output_path,
    )
