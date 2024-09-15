import pandas as pd
import numpy as np


class ObsData:
    """class to store vectors corresponding to partial lightcurves"""
    def __init__(self, data=None):
        if data is not None:
            self._obs_data = data
        else:
            self._obs_data = []
        self._data_joined = None

    def __iadd__(self, other):
        """appends partial lightcurve to the list of partial lightcurves"""
        if isinstance(other, pd.DataFrame) or isinstance(other, np.ndarray):
            self._obs_data.append(other)
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'ObsData' and '{type(other)}'")
        return self

    def __repr__(self):
        return f"ObsData (contains {self.__len__()} data blocks)"

    def __getitem__(self, index):
        """returns the partial lightcurve at the given index or slice"""
        if isinstance(index, slice):
            return ObsData(self._obs_data[index])

        return self._obs_data[index]

    def __len__(self):
        """returns the number of partial lightcurves"""
        return len(self._obs_data)

    def join_data(self):
        """joins partial lightcurves into one dataframe"""
        if isinstance(self._obs_data[0], pd.DataFrame):
            self._data_joined = pd.concat([partial_data for partial_data in self._obs_data], ignore_index=True)
        elif isinstance(self._obs_data[0], np.ndarray):
            self._data_joined = np.concatenate([partial_data for partial_data in self._obs_data])

    @property
    def joined(self):
        """returns the data of the joined lightcurves"""
        if self._data_joined is None:
            self.join_data()
            return self._data_joined
        else:
            return self._data_joined

    def save_data(self, filename):
        """saves lightcurves to a csv file"""
        self.join_data()
        if 'part_idx' not in self._data_joined.columns:
            for idx, _ in enumerate(self._obs_data):
                self._obs_data[idx]['part_idx'] = idx
        self.join_data()
        data = self.joined
        data.to_csv(filename, index=False)

    def load_data(self, filename):
        """loads lightcurves from a csv file"""
        data = pd.read_csv(filename)
        for curve_idx in data['part_idx'].unique():
            self._obs_data.append(data[data['part_idx'] == curve_idx])


class PartialLightCurve:
    """contains data for one part of a lightcurve"""

    def __init__(self, lightcurve_data: pd.DataFrame):
        self.data = lightcurve_data
        self.data['weight'] = 1.0
        self.shift = 0

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return (f"PartialLightCurve ({self.__len__()} points from {self.data['epoch'].min()} "
                f"to {self.data['epoch'].max()}), shift = {self.shift:.2f}")

    def calculate_shift(self, other: np.ndarray):
        """calculates the amount of vertical shift for the lightcurve"""
        residuals = self.data['mag'] - other
        # amount of vertical shift for the lightcurve
        self.shift = np.sum(residuals / self.data['mag_err'] ** 2) / np.sum(self.data['mag_err'] ** (-2))

    def shift_curve(self):
        """shifts the lightcurve by the calculated shift up of down"""
        self.data['mag_shifted'] = self.data['mag'].astype(float) - self.shift
        self.data['shift'] = self.shift


class LightCurve:
    """contains partial lightcurves and performs operations on them"""

    def __init__(self, lightcurve_data=None):
        if lightcurve_data is not None:
            self._lightcurves = lightcurve_data
        else:
            self._lightcurves = []
        self._lightcurves_shifted = []
        self._lightcurves_joined = None

    @property
    def lightcurves(self):
        return self._lightcurves

    def __iadd__(self, other):
        """appends partial lightcurve to the list of partial lightcurves"""
        if isinstance(other, PartialLightCurve):
            self._lightcurves.append(other)
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'LightCurve' and '{type(other)}'")
        return self

    def __len__(self):
        """returns the number of partial lightcurves"""
        return len(self._lightcurves)

    def __getitem__(self, index):
        """returns the partial lightcurve at the given index or slice"""
        if isinstance(index, slice):
            return LightCurve(self._lightcurves[index])
        return self._lightcurves[index]

    def __repr__(self):
        return f"LightCurve (contains {self.__len__()} partial lightcurves)"

    def __iter__(self):
        return iter(self._lightcurves)

    def __getitem__(self, index):
        """returns the partial lightcurve at the given index"""
        return self._lightcurves[index]

    def sort_lightcurves(self):
        """sorts partial lightcurves by their epochs in ascending order"""
        self._lightcurves = sorted([lc for lc in self.lightcurves], key=lambda x: x.data['epoch'].min())
        for curve_idx, lc in enumerate(self._lightcurves):
            lc.data['curve_idx'] = curve_idx
        self._lightcurves_joined = None

    def join_lightcurves(self):
        """joins partial lightcurves into one dataframe"""
        self._lightcurves_joined = pd.concat([partial_curve.data for partial_curve in self._lightcurves],
                                             ignore_index=True)

    @property
    def joined(self):
        if self._lightcurves_joined is None:
            self.join_lightcurves()
            return self._lightcurves_joined
        else:
            return self._lightcurves_joined

    @property
    def n_points(self):
        if self._lightcurves_joined is None:
            self.join_lightcurves()
            return len(self._lightcurves_joined)
        else:
            return len(self._lightcurves_joined)

    @property
    def shifts(self):
        """get value of the sifts for every partial lightcurve"""
        return [curve.shift for curve in self._lightcurves]

    def calculate_shifts(self, other: ObsData):
        """calculates the amount of vertical shift for every partial lightcurve"""
        for idx, curve in enumerate(self._lightcurves):
            curve.calculate_shift(other[idx])

    def shift_curves(self):
        """shifts all the partial lightcurves"""
        for curve in self._lightcurves:
            curve.shift_curve()
        self._lightcurves_joined = None

    def enum_curves(self):
        for idx, curve in enumerate(self._lightcurves):
            print(f'{idx}. {curve}')

    def save_data(self, filename):
        """saves lightcurves to a csv file"""
        data = self.joined[['epoch', 'mag', 'mag_err', 'weight', 'curve_idx']]
        data.to_csv(filename, index=False)

    def load_data(self, filename):
        """loads lightcurves from a csv file"""
        data = pd.read_csv(filename)
        for curve_idx in data['curve_idx'].unique():
            self._lightcurves.append(PartialLightCurve(data[data['curve_idx'] == curve_idx]))

    def update_parts(self, data: pd.DataFrame):
        """updates ALL partial lightcurves with the provided data"""
        for curve in self._lightcurves:
            # get the idx of the current partial lightcurve
            curve_idx = curve.data['curve_idx'].iloc[0]
            # update the partial lightcurve data with the provided data
            curve.data = data[data['curve_idx'] == curve_idx]



