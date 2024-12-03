import json
import logging
import os
import re
import warnings
from glob import glob
from types import NoneType
from typing import Union, Dict, List, Iterator

import numpy
import numpy as np
import pandas
from sklearn.model_selection import train_test_split

from src.constants import SAMPLE_MANIFESTS_DIR, DATA_RESOURCES_DIR, PTM_LABEL_ENUM


logging.basicConfig()
logger = logging.getLogger(__name__)


class ManifestBase(pandas.DataFrame):
    """
    Implements a general class with metadata fields to attach to the
    main dataframe.
    Try not to add more methods to this class. Simply inherit from it and add
    your methods and features to that class.
    """

    _metadata = []
    _required_metadata = []

    @property
    def _constructor(self):
        def f(*args, **kwargs):
            for attr_name in self._metadata:
                kwargs[attr_name] = getattr(self, attr_name)
            return type(self)(*args, **kwargs)

        return f

    def __init__(self, data: Union[pandas.DataFrame, Dict], *args, **kwargs):
        """
        Initializes a dataframe from data_resources
        :param data:
        :param args:
        :param kwargs:
        """
        # ensure that required_metadata is a subset of metadata
        assert all(x in self._metadata for x in self._required_metadata)

        # pandas doesn't call this internal _constructor,
        # making this necessary.
        if not isinstance(data, pandas.core.internals.BlockManager):
            for attr_name in self._required_metadata:
                if attr_name not in kwargs:
                    raise ValueError(f"arg '{attr_name}' is required")

        # set the metadata args
        for attr_name in self._metadata:
            # This prevents attr_name from being passed to
            # ManifestBase constructor
            setattr(self, attr_name, kwargs.pop(attr_name, None))

        super().__init__(data, *args, **kwargs)


class Sample:
    """
    This class defines the requirements

    This class facilitates fetching data_resources in a given region and also makes sure the given regions are valid.
    """

    def __init__(
        self,
        run_id: str,
        sample_id: str,
        biomaterial: str,
        assay_type: str,
        sample_name: Union[NoneType, str] = None,
        metadata: Union[NoneType, str] = None,
        *args,
        **kwargs,
    ):
        """

        :param run_id:
        :param sample_id:
        :param biomaterial:
        :param assay_type:
        :param sample_name:
        :param metadata:
        :param args:
        :param kwargs:
        """

        self.run_id = run_id
        self.sample_id = sample_id
        self.biomaterial = biomaterial
        self.assay_type = assay_type
        self.sample_name = sample_name
        self.metadata = metadata

        self.is_valid()

    def is_valid(self) -> None:

        assert self.assay_type in [
            "WGBS",
            "WES",
            "WGS",
        ], f"assay_type must be in ['WGBS', 'WES', 'WGS']"

    @classmethod
    def from_series(cls, series: pandas.Series) -> "Sample":

        return cls(
            run_id=series.run_id,
            sample_id=series.sample_id,
            biomaterial=series.biomaterial,
            assay_type=series.assay_type,
            sample_name=series.sample_name,
            metadata=json.dumps(series.to_dict()),
        )

    @property
    def id(self) -> str:
        return self.run_id

    def __str__(self) -> str:
        return self.run_id

    def to_dict(self):
        return dict(
            run_id=self.run_id,
            sample_id=self.sample_id,
            biomaterial=self.biomaterial,
            assay_type=self.assay_type,
            sample_name=self.sample_name,
        )

    def __eq__(self, other: "Sample"):

        if self.run_id == other.run_id:
            if not (
                self.sample_id == other.sample_id
                and self.biomaterial == other.biomaterial
                and self.assay_type == other.assay_type
            ):
                raise ValueError(
                    f"Invalid sample! if run_ids match, the rest of the required information must match too."
                )
            else:
                return True
        else:
            return False


class SampleManifest(ManifestBase):
    """
    This construct is used for sample selection and information retrieval. As a standard, for a given sample set,
    there should be an accompanying sample_dataframe.epr674.tsv which contains sample information and contains
    the following columns:
        "run_id",
        "sample_id",
        "biomaterial",
        "assay_type",
        "sample_name",
        "metadata",

    run_id: is the id corresponding to the data_resources generation on a biosample and this column must have
     unique values.
    sample_id: is an id corresponding to a biological entity, e.g. human sample
    assay_type: the type of assay used in data_resources generation, e.g. WGBS, WES, etc.
    sample_name: a more human-readable name for the samples
    metadata: all other information must be stored in the form of a json.dump()

    Example:
        SampleManifest.load_sample_set(sample_set = 'alizadeh_nature_biotechnology_22')

    """

    __SAMPLE_SETS__ = {
        os.path.basename(i).split(".sm")[0]: i
        for i in glob(os.path.join(SAMPLE_MANIFESTS_DIR, "*"))
        if i.endswith(".sm")
    }

    _required_columns = [
        "run_id",
        "sample_id",
        "sample_name",
        "biomaterial",
        "assay_type",
        "metadata",
    ]

    _optional_columns = [
        "age",
        "sex",
        "cfdna_concentration",
        "input_mass",
        "url",
    ]
    _standard_columns = _required_columns + _optional_columns

    def __init__(self, data, *args, **kwargs):
        if (
            len(data) == 0
            and hasattr(data, "columns")
            and len(set(self._required_columns) - set(data.columns)) > 0
        ):
            # Handle empty dataframes by imposing null columns
            data = pandas.DataFrame(
                columns=self._standard_columns + self._required_columns
            )

        super().__init__(data, *args, **kwargs)

        if isinstance(data, pandas.core.internals.BlockManager):
            return

        for column in self._required_columns:
            if column not in self.columns:
                raise KeyError(
                    f"Required column {column} is missing. Must contain {self._required_columns} columns."
                )

        # if len(self["run_id"]) != len(set(self["run_id"])):
        #     raise ValueError("duplicate run_ids")

    def __len__(self):
        return len(self.index)

    def __eq__(self, other: "SampleManifest") -> bool:
        """
        Equality operand that checks if self and other are equal

        :param other: RegionManifest
        :return: bool
        """

        if len(self) != len(other):
            return False

        if (self.run_id.values != other.run_id.values).all():
            return False

        if (self.assay_type.values != other.assay_type.values).all():
            return False

        return True

    def __add__(self, other: "SampleManifest") -> bool:
        sample_dataframe = pandas.concat([self, other], ignore_index=True, join="inner")

        return type(self)(sample_dataframe)

    @classmethod
    def load_sample_set(cls, sample_set: str) -> "SampleManifest":
        """
        Loads a sample set from a sample_dataframe.csv
        :param sample_set: sample_set name
        :return: SampleManifest
        """

        if sample_set not in cls.__SAMPLE_SETS__:
            raise ValueError(
                f"sample_set {sample_set} is not valid, choose one of {list(SampleManifest.__SAMPLE_SETS__.keys())}"
            )
        else:
            sample_dataframe_tsv = SampleManifest.__SAMPLE_SETS__[sample_set]
            if not os.path.isfile(sample_dataframe_tsv):
                raise FileNotFoundError(f"{sample_dataframe_tsv} does not exist!")

        return cls(pandas.read_table(sample_dataframe_tsv))

    def itersamples(self) -> Iterator:

        for name, row in self.iterrows():
            sample = Sample(
                run_id=row.run_id,
                sample_id=row.sample_id,
                sample_name=row.sample_name,
                biomaterial=row.biomaterial,
                assay_type=row.assay_type,
                metadata=row.metadata,
            )
            yield sample

    def update_metadata(
        self, values: Union[List, pandas.Series], name: Union[str, None] = None
    ) -> "SampleManifest":
        """
        Updates the metadata columns of a dataframe
        :param values: vector of values to be updated
        :param name: name of the metadata column
        :return: MethylationSampleDataFrame
        """

        if len(values) != self.shape[0]:
            raise ValueError(
                f"Can not set column of size {self.shape[0]} with a vector of size {len(values)}"
            )

        if "metadata" not in self.columns or all(self["metadata"].unique() == ["none"]):
            self["metadata"] = json.dumps({})

        if isinstance(values, pandas.Series):
            name = values.name if name is None else name
            if name in self.columns:
                raise ValueError(f"Column {name} already exists.")
            else:
                values = values.rename(name)

            if set(values.index) != set(self["run_id"]):
                raise ValueError("Values must match 1to1 to the run_id")
            self.loc[:, name] = values.loc[self["run_id"]].values

        elif isinstance(values, Union[List, numpy.ndarray]):
            warnings.warn("Make sure the order or values are correct!")

            if name is None:
                raise ValueError("If 'values' is a list, you must pass a name")
            elif name in self.columns:
                raise ValueError(f"Column {name} already exists.")

            self.loc[:, name] = values
        else:
            raise TypeError(f"Can not accept values of type {type(values)}")

        def update_return_dict(dict_: Dict, dict_n: Dict):
            """
            updates a dictionary and returns a dictionary rather than updating inplace
            :param dict_: dict
            :param dict_n: dict
            :return: dict
            """
            dict_.update(dict_n)

            return dict_

        self["metadata"] = self.apply(
            lambda x: json.dumps(
                update_return_dict(json.loads(x["metadata"]), {name: x[name]})
            ),
            axis=1,
        )

        return self.drop(name, axis=1)

    @property
    def run_ids(self):
        return self.run_id.values

    def attach_metadata(self) -> "SampleManifest":
        """
        Unpacks, converts and attaches the json string in 'metadata' column and returns
        an instance of the same class
        :return: MethylationSampleDataFrame
        """
        if "metadata" not in self.columns:
            raise KeyError(f"metadata must be in columns")

        metadata = pandas.DataFrame.from_records(
            self["metadata"].apply(lambda val: json.loads(val)).values
        )

        return type(self)(pandas.concat([self, metadata], axis=1))

    def saveas(self, fpath: str):
        """
        Saves the sample dataframe in the correct format
        :param fpath: output path
        :return: None
        """

        self.to_csv(fpath, sep="\t", header=True, index=False)

    def train_test_split(
        self,
        use_train_test_columns=True,
        test_size=None,
        train_size=None,
        random_state=None,
        shuffle=True,
        stratify=None,
    ):
        """

        :param use_train_test_columns:
        :param test_size:
        :param train_size:
        :param random_state:
        :param shuffle:
        :param stratify:
        :return:
        """

        if use_train_test_columns:
            X_train = self.attach_metadata().query("train_test == 'train'")
            X_test = self.attach_metadata().query("train_test == 'test'")
        else:

            data = self.attach_metadata()
            sm_nci = data.query("dataset=='NCI'")
            sm_other = data.query("dataset!='NCI'")

            if test_size is None:
                test_size = sm_nci.query("train_test == 'test'").shape[0]

            X_train, X_test = train_test_split(
                sm_nci,
                test_size=test_size,
                train_size=train_size,
                random_state=random_state,
                shuffle=shuffle,
                stratify=stratify,
            )
            X_train = type(self)(X_train)
            X_test = type(self)(pandas.concat([X_test, sm_other], axis=0))

        return X_train.reset_index(drop=True), X_test.reset_index(drop=True)


class AntigenSampleManifest(SampleManifest):

    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)

    def fetch_data(self):
        """

        :return:
        """

        if "fpath" not in self.columns:
            data = self.copy().attach_metadata()
        else:
            data = self.copy()

        data = pandas.concat(
            data.apply(
                lambda x: pandas.read_table(
                    os.path.join(DATA_RESOURCES_DIR, x["fpath"]), sep="\t"
                ).assign(
                    run_id=x["run_id"],
                    dataset=x["dataset"],
                ),
                axis=1,
            ).values
        )

        return data


class AntigenDataset:

    def __init__(self, data: pandas.DataFrame, sample_set: str):

        self.sample_set = sample_set
        self.data = data
        if not all(
            [c in data.columns for c in ["response_type", "sequence_id", "run_id"]]
        ):
            raise KeyError(f"Invalid dataset")

    @classmethod
    def load_neopeptide_dataset(cls):

        data = AntigenSampleManifest.load_sample_set("neopep").fetch_data()

        return cls(data, "neopep")

    @classmethod
    def load_mutation_dataset(cls):
        data = AntigenSampleManifest.load_sample_set("mutation").fetch_data()

        return cls(data, "mutation")

    def write_fasta_sequences(
        self, output_dir: str, columns: Union[str, NoneType] = None
    ):

        if not os.path.isdir(output_dir):
            raise NotADirectoryError(f"Directory does not exist!")

        if columns is None:
            columns = ["wt_seq", "mutant_seq"]

        for column in columns:

            sequences = (
                self.data[["sequence_id", column]]
                .drop_duplicates()
                .set_index("sequence_id")
                .to_dict()[column]
            )

            with open(os.path.join(output_dir, f"{column}.fasta"), "w") as fasta_file:
                for name, seq in sequences.items():
                    fasta_file.write(f">{name}\n")
                    fasta_file.write(f"{seq}\n")

            logger.info(f"Wrote {column}.fasta")

    @staticmethod
    def parse_sequence_id(sequence_id: str, seq_len: int = None):
        """

        :param sequence_id:
        :param seq_len:
        :return:
        """

        idx = list(re.finditer(re.escape("_"), sequence_id))

        sid = sequence_id[: idx[-3].span()[0]]
        label = sequence_id[idx[-2].span()[1] :]
        label_index = int(PTM_LABEL_ENUM[label])
        site = int(sequence_id[idx[-3].span()[1] : idx[-2].span()[0]])
        seq = sequence_id[idx[1].span()[1] : idx[2].span()[0]]

        return dict(
            sid=sid,
            label=label,
            label_index=label_index,
            site=site - 1,
            seq_len=len(seq) if seq_len is None else seq_len,
        )

    @staticmethod
    def parse_mind_json(fpath: str) -> Dict[str, numpy.ndarray]:
        """

        :param fpath:
        :return:
        """

        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"File {fpath} does not exist!")
        if os.path.splitext(fpath)[1] != ".json":
            raise TypeError(f".json file type expected.")

        with open(fpath, "r") as f_j:
            contents = json.loads(f_j.read())

        prob_matrices = {}
        for k, v in contents.items():
            k_parsed = AntigenDataset.parse_sequence_id(sequence_id=k)

            if k_parsed["sid"] not in prob_matrices:
                prob_matrices[k_parsed["sid"]] = numpy.zeros(
                    [len(PTM_LABEL_ENUM), k_parsed["seq_len"]]
                )

            prob_matrices[k_parsed["sid"]][
                k_parsed["label_index"], k_parsed["site"]
            ] = float(v)

        return prob_matrices

    @staticmethod
    def mind_json_to_dataframe(
        fpath: str, name: str = None, reduction: str = "mean", axis: int = 1
    ):
        """

        :param fpath:
        :param name:
        :param reduction:
        :param axis:
        :return:
        """

        label_ptm = [(v, k) for k, v in PTM_LABEL_ENUM.items()]
        label_ptm.sort()

        pproba_matrices = AntigenDataset.parse_mind_json(fpath=fpath)
        pproba_matrices = pandas.Series(pproba_matrices).rename(
            name if name is not None else os.path.basename(fpath)
        )
        index = pproba_matrices.index

        if reduction is None:
            return pproba_matrices
        elif reduction == "mean":
            pproba_matrices = pproba_matrices.apply(lambda x: x.mean(axis=axis))
        else:
            raise ValueError(f"only mean is accepted for now!")

        pproba_matrices = pandas.DataFrame(np.array(pproba_matrices.values.tolist()))
        pproba_matrices.index = index
        if axis == 1:
            pproba_matrices.columns = [i[1] for i in label_ptm]

        return pproba_matrices

    @staticmethod
    def split_to_id_and_data(data: pandas.DataFrame):

        id_columns = [
            "dataset",
            "Nb_Samples",
            "TumorContent",
            # "Zygosity",
            "aa_mutant",
            "aa_wt",
            "alt",
            "chromosome",
            "gene",
            "genomic_coord",
            "mutant_seq",
            "pep_mut_start",
            "protein_coord",
            "ref",
            "response_type",
            "run_id",
            "sequence_id",
            "wt_seq",
            "mutation_type",
            "mutant_best_alleles",
            "wt_best_alleles",
            "mutant_best_alleles_netMHCpan",
            "mutant_other_significant_alleles_netMHCpan",
            "wt_best_alleles_netMHCpan",
        ]

        data_id = data[[c for c in data.columns if c in id_columns]]
        data_data = data[[c for c in data.columns if c not in id_columns]]

        return data_id, data_data

    @staticmethod
    def construct_datasets(
        sample_set: str = "mutation",
        supervised: bool = True,
        **train_test_split_kwargs,
    ):

        sm = AntigenSampleManifest.load_sample_set(sample_set)
        sm_train, sm_test = sm.train_test_split(**train_test_split_kwargs)
        X_train = sm_train.fetch_data()
        X_test = sm_test.fetch_data()

        Xtr_id, X_train = AntigenDataset.split_to_id_and_data(X_train)
        idx_tr = Xtr_id.apply(
            lambda x: "__".join(x[["run_id", "sequence_id"]]), axis=1
        ).values
        X_train.index = idx_tr
        Xtr_id.index = idx_tr

        Xte_id, X_test = AntigenDataset.split_to_id_and_data(X_test)
        idx_te = Xte_id.apply(
            lambda x: "__".join(x[["run_id", "sequence_id"]]), axis=1
        ).values
        X_test.index = idx_te
        Xte_id.index = idx_te

        if supervised:

            label_dict = {"negative": 0, "CD8": 1}

        else:

            label_dict = {"negative": 0, "CD8": 1, "untested": np.NaN}

        idx_tr = Xtr_id["response_type"].apply(lambda x: x in label_dict.keys()).values
        X_train = X_train.loc[idx_tr]
        Xtr_id = Xtr_id.loc[idx_tr]

        train_info = Xtr_id[
            [
                "run_id",
                "response_type",
                "dataset",
            ]
        ]
        y_train = Xtr_id["response_type"].map(label_dict).values

        idx_te = Xte_id["response_type"].apply(lambda x: x in label_dict.keys()).values
        X_test = X_test.loc[idx_te]
        Xte_id = Xte_id.loc[idx_te]

        test_info = Xte_id[
            [
                "run_id",
                "response_type",
                "dataset",
            ]
        ]
        y_test = Xte_id["response_type"].map(label_dict).values

        return dict(
            train=(X_train, y_train, train_info),
            test=(X_test, y_test, test_info),
        )
