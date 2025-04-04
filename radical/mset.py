from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
from casacore.tables import table, taql, makecoldesc
import numpy as np

class MS(object):
    def __init__(self, path):
        tbl = table(path, ack=True, readonly=False)

        self.tbl = taql("select * from $tbl where ANTENNA1 <> ANTENNA2 and !FLAG_ROW")
        print(f"Found {self.tbl.nrows()} rows")

    @property
    def shape(self):
        return (self.tbl.nrows(), self.tbl.getcoldesc("DATA")["shape"][0], 2, 2)

    @property
    def freq(self):
        return self.tbl.SPECTRAL_WINDOW.getcell("CHAN_FREQ", 0)

    @property
    def wavelength(self):
        return 299_792_458 / self.freq

    @property
    def uvw(self):
        return self.tbl.getcol("UVW")

    @property
    def data(self):
        data = self.tbl.getcol("DATA")
        return data.reshape(data.shape[0], data.shape[1], 2, 2)

    @data.setter
    def data(self, corrected):
        shape = self.tbl.getcoldesc("DATA")["shape"]
        self.tbl.putcol("DATA", corrected.reshape(self.tbl.nrows(), *shape))
        self.tbl.flush()

    @property
    def corrected(self):
        corrected = self.tbl.getcol("CORRECTED_DATA")
        return corrected.reshape(corrected.shape[0], corrected.shape[1], 2, 2)

    @corrected.setter
    def corrected(self, corrected):
        if "CORRECTED_DATA" not in self.tbl.colnames():
            print("Creating CORRECTED_DATA column")
            self.tbl.addcols(makecoldesc("CORRECTED_DATA", self.tbl.getcoldesc("DATA")))

        shape = self.tbl.getcoldesc("CORRECTED_DATA")["shape"]
        self.tbl.putcol("CORRECTED_DATA", corrected.reshape(self.tbl.nrows(), *shape))
        self.tbl.flush()

    @property
    def model(self):
        model = self.tbl.getcol("MODEL_DATA")
        return model.reshape(model.shape[0], model.shape[1], 2, 2)

    @model.setter
    def model(self, model):
        if "MODEL_DATA" not in self.tbl.colnames():
            print("Creating MODEL_DATA column")
            self.tbl.addcols(makecoldesc("MODEL_DATA", self.tbl.getcoldesc("DATA")))

        shape = self.tbl.getcoldesc("MODEL_DATA")["shape"]
        self.tbl.putcol("MODEL_DATA", model.reshape(self.tbl.nrows(), *shape))
        self.tbl.flush()

    @property
    def weight(self):
        weight = self.tbl.getcol("WEIGHT_SPECTRUM")
        weight *= self.tbl.getcol("WEIGHT")[:, None, :]
        weight *= ~self.tbl.getcol("FLAG")
        return weight.reshape(weight.shape[0], weight.shape[1], 2, 2)

    @property
    def antenna1(self):
        return self.tbl.getcol("ANTENNA1")

    @property
    def antenna2(self):
        return self.tbl.getcol("ANTENNA2")

    @property
    def midfreq(self):
        return np.mean(self.freq)

    @property
    def mwadelays(self):
        return self.tbl.MWA_TILE_POINTING.getcell("DELAYS", 0)

    @property
    def phasecenter(self):
        return SkyCoord(*self.tbl.FIELD.getcell("PHASE_DIR", 0)[0], unit=("rad", "rad"))

    @property
    def midtime(self):
        # This time is based on the weights of each row
        return Time(np.mean(np.unique(self.tbl.getcol("TIME_CENTROID"))) / (24 * 60 * 60), format="mjd")
        print("Other midtime:", np.mean(np.unique(self.tbl.getcol("TIME_CENTROID"))) / (24 * 60 * 60))
        times = self.tbl.getcol("TIME_CENTROID") * self.weight.sum(axis=(1, 2))
        time = Time(times.sum() / self.weight.sum() / (24 * 60 * 60), format="mjd")
        print("Midtime:", time)
        return time

    @property
    def midlocation(self):
        # TODO: weight each antenna by its use
        x, y, z = np.mean(self.tbl.ANTENNA.getcol("POSITION"), axis=0)
        return EarthLocation.from_geocentric(x, y, z, "meter")

    @property
    def nants(self):
        return max(self.antenna1.max(), self.antenna2.max()) + 1

    def antennas(self, ref=0):
        """
        Get positions of antennas with respect to the reference antenna.
        """
        # Get antenna locations (XYZ coordinates)
        ants = self.tbl.ANTENNA.getcol("POSITION")

        # Compute longitude and lattitude of reference antenna
        r = np.sqrt(np.sum(ants[ref]**2))
        theta = np.arccos(ants[ref][2] / r)
        phi = np.arctan2(ants[ref][1], ants[ref][0])

        # Construct rotation matrices such that (r2 x r1) would
        # rotate the North celestial pole to the reference antenna
        # location

        # First: rotate around the y axis
        r1 = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        # Then around the z axis
        r2 = np.array([
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi), np.cos(phi), 0],
            [0, 0, 1]
        ])

        # Now we apply the inverse rotation---rotate the reference antenna
        # to the pole---and thus find the relative  coordinates of each antenna
        return (np.linalg.inv(r2 @ r1) @ ants.reshape((-1, 3, 1))).reshape((-1, 3))