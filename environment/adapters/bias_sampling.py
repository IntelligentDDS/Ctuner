import io
import logging
from functools import partial
from typing import Optional, Union, List

import ConfigSpace as CS
import numpy as np

logger = logging.getLogger(__name__)

KNOBS_WITH_SPECIAL_VALUES = {
    'autovacuum_vacuum_cost_delay': {
        # Value of -1 infers the value from `vacuum_cost_delay'
        'subgrid': 'autovacuum',
        'special_value': -1,
    },
    'autovacuum_vacuum_cost_limit': {
        # Value of -1 infers the value from `vacuum_cost_limit' (default)
        'subgrid': 'autovacuum',
        'special_value': -1,
    },
    'autovacuum_work_mem': {
        # Value of -1 infers the value from `maintenance_work_mem' param (default)
        'subgrid': 'autovacuum',
        'special_value': -1,
    },
    'backend_flush_after': {
        # Value of 0 disables forced writeback (default)
        'special_value': 0,
    },
    'bgwriter_flush_after': {
        # Value of 0 disables forced writeback
        'subgrid': 'bgwriter',
        'special_value': 0,
    },
    'bgwriter_lru_maxpages': {
        # Value of 0 disables background writing
        'subgrid': True,
        'special_value': 0,
    },
    'checkpoint_flush_after': {
        # Value of 0, disables forced writeback (default)
        'special_value': 0,
    },
    'effective_io_concurrency': {
        # Value of 0 disables issuance of asynchronous I/O requests
        'special_value': 0,
    },
    'geqo_generations': {
        # Value will be infered from `geqo_effort' knob (default)
        'subgrid': 'geqo',
        'special_value': 0,
    },
    'geqo_pool_size': {
        # Value will be infered from `geqo_effort' knob (default)
        'subgrid': 'geqo',
        'special_value': 0,
    },
    'max_parallel_workers_per_gather': {
        # Value of 0 disables parallel query execution (default)
        'special_value': 0,
    },
    'max_prepared_transactions': {
        # Value of 0 disables the prepared-transaction feature (default)
        'special_value': 0,
    },
    'old_snapshot_threshold': {
        # Value of -1 disables the feature (default)
        'special_value': -1,
    },
    'temp_file_limit': {
        # Value of -1 means no limit
        'special_value': -1,
    },
    'vacuum_cost_delay': {
        # Value of 0, disables the cost-based vacuum delay feature (default)
        'subgrid': True,
        'special_value': 0,
    },
    'wal_buffers': {
        # Value of 1 infers value from `shared_buffers' (default)
        # More specifically: wal_buffers = 1/32 * shared_buffers
        'special_value': -1,
    },
    'wal_writer_flush_after': {
        # Value of 0 forces WAL data to be flushed immediately
        'special_value': 0,
    },
}

def special_value_scaler(hp, value):
    if value < hp._special_value_prob:
        # Fix value to special value
        return hp._inverse_transform(hp._special_value)
    # uniformly scale remaining percentage to all other values
    return hp._inverse_transform(hp._transform_scalar(
        (value - hp._special_value_prob) / (1. - hp._special_value_prob)))


class UniformIntegerHyperparameterWithSpecialValue(CS.UniformIntegerHyperparameter):
    def __init__(self, *args, special_value: Optional[int] = None,
                            special_value_prob: Optional[float], **kwargs):
        super().__init__(*args, **kwargs)

        assert special_value >= self.lower and special_value <= self.upper, \
            ('Special value [=%d] should be inside the range' % special_value)
        assert special_value_prob > 0 and special_value_prob < 1, \
            ('Probability for special value should be in (0, 1)')
        assert special_value == self.lower, \
            ('For now implementation supports only special value to be on the lower end')

        self._special_value = special_value
        self._special_value_prob = special_value_prob

    def _sample(self, rs: np.random.RandomState, size: Optional[int] = None
                ) -> Union[np.ndarray, float]:

        # Bias sample on the special value
        samples = rs.uniform(size=size)

        if size is None:
            return special_value_scaler(self, value)

        special_value_scaler_vector = partial(special_value_scaler, self)
        return np.array(list(map(special_value_scaler_vector, samples)))

    def __repr__(self) -> str:
        repr_str = io.StringIO()
        repr_str.write("%s, Type: UniformIntegerWithSpecialValues, Range: [%s, %s], Default: %s"
                       % (self.name, repr(self.lower),
                          repr(self.upper), repr(self.default_value)))
        if self.log:
            repr_str.write(", on log-scale")
        if self.q is not None:
            repr_str.write(", Q: %s" % repr(self.q))
        repr_str.seek(0)
        return repr_str.getvalue()


class PostgresBiasSampling:
    def __init__(self, adaptee: CS.ConfigurationSpace, seed: int, bias_prob_sv: float):
        self._adaptee: CS.ConfigurationSpace = adaptee
        self._target: CS.ConfigurationSpace = None
        self._seed: int = seed

        assert bias_prob_sv > 0 and bias_prob_sv < 1, \
            'Bias on the special values should be in (0, 1) range'
        self._bias_prob_sv: float = bias_prob_sv

        self._build_biased_config_space()

    def _build_biased_config_space(self):
        root_hyperparams = [ ]
        for adaptee_hp in self.adaptee.get_hyperparameters():

            if adaptee_hp.name not in KNOBS_WITH_SPECIAL_VALUES.keys():
                # no need to bias sampling
                root_hyperparams.append(adaptee_hp)
                continue

            # NOTE: assume splitting occurs in the lower point
            info = KNOBS_WITH_SPECIAL_VALUES[adaptee_hp.name]
            assert isinstance(adaptee_hp, CS.UniformIntegerHyperparameter)
            assert adaptee_hp.lower == info['special_value']

            biased_hyperparam = UniformIntegerHyperparameterWithSpecialValue(
                adaptee_hp.name, adaptee_hp.lower, adaptee_hp.upper,
                default_value=adaptee_hp.default_value,
                special_value=info['special_value'],
                special_value_prob=self._bias_prob_sv,
            )
            root_hyperparams.append(biased_hyperparam)

        root = CS.ConfigurationSpace(
            name=self._adaptee.name,
            seed=self._seed,
        )
        root.add_hyperparameters(root_hyperparams)

        self._target = root

    @property
    def adaptee(self) -> CS.ConfigurationSpace:
        return self._adaptee

    @property
    def target(self) -> CS.ConfigurationSpace:
        return self._target

    def project_point(self, point):
        raise NotImplementedError()

    def unproject_point(self, point):
        coords = point.get_dictionary()
        valid_dim_names = [ dim.name for dim in self.adaptee.get_hyperparameters() ]
        assert all(dim in valid_dim_names for dim in coords.keys())
        return coords

    def project_dataframe(self, df, in_place: bool):
        raise NotImplementedError

    def unproject_dataframe(self, df, in_place: bool):
        raise NotImplementedError
