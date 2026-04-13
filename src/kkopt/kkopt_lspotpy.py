# kkopt_lspotpy.py
import numpy as np


class lspotpy_object_factories(object):
    def __init__(self, _kinds):
        self._kinds = _kinds
        self.factories = dict()

    @property
    def kinds(self):
        return self._kinds

    def register(self, _kind, _obj):
        self.factories[_kind] = _obj

    def create(self, _kind, **_kwargs):
        if _kind in self.factories:
            return self.factories[_kind].create(_kwargs)
        return None


LSPOTPY_INTERPOLATION_FACTORIES = lspotpy_object_factories("interpolators")


class lspotpy_interpolation(object):
    def __init__(self):
        pass

    def interpolate(self, _num):
        raise NotImplementedError("missing implementation")


class lspotpy_interpolation_factory(object):
    def __init__(self, _kind):
        self.kind = _kind
        # register factory
        if self.kind not in LSPOTPY_INTERPOLATION_FACTORIES.factories:
            LSPOTPY_INTERPOLATION_FACTORIES.register(self.kind, self)


class lspotpy_interpolation_linear(lspotpy_interpolation):
    def __init__(self, minvalue=np.nan, maxvalue=np.nan):
        super(lspotpy_interpolation_linear, self).__init__()
        self.minvalue = minvalue
        self.maxvalue = maxvalue

    def interpolate(self, _num):
        return list(np.linspace(self.minvalue, self.maxvalue, _num))


class lspotpy_interpolation_factory_linear(lspotpy_interpolation_factory):
    def __init__(self, _kind="linear"):
        super(lspotpy_interpolation_factory_linear, self).__init__(_kind)

    def create(self, **_kwargs):
        return lspotpy_interpolation_linear(**_kwargs)


# Instantiate the linear interpolation factory so it is registered
_ = lspotpy_interpolation_factory_linear("linear")


class lspotpy_distribution(object):
    def __init__(self, _kind, **_kwargs):
        self.kind = _kind
        self.minvalues = None
        self.maxvalues = None
        self.values = None
        self.interpolation = None

        if "minvalues" in _kwargs:
            self.minvalues = _kwargs["minvalues"]
            if not isinstance(self.minvalues, list):
                self.minvalues = [self.minvalues]
        if "maxvalues" in _kwargs:
            self.maxvalues = _kwargs["maxvalues"]
            if not isinstance(self.maxvalues, list):
                self.maxvalues = [self.maxvalues]
        if "values" in _kwargs:
            if self.minvalues is not None or self.maxvalues is not None:
                raise RuntimeError(
                    'arguments "minvalues", "maxvalues" or "values" are mutually exclusive'
                )
            self.values = _kwargs["values"]
        if "interpolation" in _kwargs:
            self.interpolation = LSPOTPY_INTERPOLATION_FACTORIES.create(
                _kwargs["interpolation"]
            )
        if self.kind == "uniform":
            self.distribution = np.random.uniform

    def sample_continuum(self):
        samples = []
        for mi, ma in zip(self.minvalues, self.maxvalues):
            samples.append(self.distribution(mi, ma))
        return samples

    def sample(self, _num):
        samples = []
        for _ in range(_num):
            samples.append(self.sample_continuum())
        return samples


def lspotpy_makedistribution(_kind, **_kwargs):
    # Depending on your original intent, this may want to use interpolation factories
    # or just directly build a distribution; here we keep the original idea minimal.
    return lspotpy_distribution(_kind, **_kwargs)


class lspotpy_parameter(object):
    def __init__(self, _name, _distribution, _properties):
        self.name = _name
        self.values = []
        self.minvalue = np.nan
        self.maxvalue = np.nan
        self.initialvalue = np.nan
        self.n_values = 0
        self.distribution = lspotpy_makedistribution(_distribution, **_properties)

    def sample(self, n_samples):
        return self.distribution.sample(n_samples)
