import os

REFERENCE_ASSEMBLIES = {"hg19", "hg38"}
DEFAULT_ASSEMBLY = "hg19"
DEFAULT_DNANEXUS_PROJECT = ""

NUCLEOTIDE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 4}
INT_TO_NUCLEOTIDE = {v: k for k, v in NUCLEOTIDE_TO_INT.items()}


REPO_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[
    0
]  # e.g. /data_resources/home/osolari/workspace/src

# Keep reference related data_resources in this directory
REFERENCE_DIR = f"{REPO_DIR}/reference/"
# data_resources where public data_resources is kept
DATA_RESOURCES_DIR = os.path.join(REPO_DIR, "data_resources")
# Test directory
TEST_DIR = f"{REPO_DIR}/src/test/"


MANIFESTS_DIR = os.path.join(REPO_DIR, "manifests")
SAMPLE_MANIFESTS_DIR = os.path.join(MANIFESTS_DIR, "samples")


ALL_AAS = "ACDEFGHIKLMNPQRSTUVWXY"
ADDITIONAL_TOKENS = ["<OTHER>", "<START>", "<END>", "<PAD>", "<MASK>"]
# MLM_ADDITIONAL_TOKENS = ['<OTHER>', '<START>', '<END>', '<PAD>','<MASK>']

# Each sequence is added <START> and <END> tokens
ADDED_TOKENS_PER_SEQ = 2

n_aas = len(ALL_AAS)
aa_to_token_index = {aa: i for i, aa in enumerate(ALL_AAS)}
additional_token_to_index = {
    token: i + n_aas for i, token in enumerate(ADDITIONAL_TOKENS)
}
token_to_index = {**aa_to_token_index, **additional_token_to_index}
index_to_token = {index: token for token, index in token_to_index.items()}
n_tokens = len(token_to_index)


PTM_LABEL_TO_AA = {
    "Hydro_K": "K",
    "Hydro_P": "P",
    "Methy_K": "K",
    "Methy_R": "R",
    "N6-ace_K": "K",
    "Palm_C": "C",
    "Phos_ST": "ST",
    "Phos_Y": "Y",
    "Pyro_Q": "Q",
    "SUMO_K": "K",
    "Ubi_K": "K",
    "glyco_N": "N",
    "glyco_ST": "ST",
}

PTM_LABELS = list(PTM_LABEL_TO_AA.keys())
PTM_LABEL_ENUM = {str(label): i for i, label in enumerate(sorted(set(PTM_LABELS)))}
