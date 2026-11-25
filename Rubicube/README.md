# RubiCube

Auth: Cris Zilleruelo (cristobal.zilleruelo.canas@desy.de)


This fodler contains all the code developement for the neutrino counterpart search. It was built on the previous healpix infrastructure in AMPEL. 

### Key AMPEL unis
Modified: 
* ampel/contrib/hu/util/AmpelHealpix.py
* ampel/contrib/hu/t4/HealpixTokenGenerator.py
* ampel_conf.yaml:
    * ``` 
        ampel-catalogmatch/url: https://ampel-ztf.zeuthen.desy.de/api/catalogmatch/
        ampel-ztf/archive: https://ampel-ztf.zeuthen.desy.de/api/ztf/archive/v3/ 
   
* pyproject.toml
    * ``` 
        rubicube = [
        "mhealpy",
        "astropy_healpix",
        "snpy (>=2.7.0,<3.0.0)",
        "lasair (>=0.1.2,<0.2.0)",
        "ligo.skymap",
        ]
   

Added:
* ampel/contrib/hu/t3/IceCubePlotTransientLightcurves.py


### Extra code

#### Neutrino and transient processing files inside RubiCube/
* ic_processing.py
    * General IC healpix map processing
* ic_visualization.py
    * General IC map plotting 
* ic_yaml.py
    * Methods for writing an AMPEL job file that searches for active transients from a neutrino alert
* params.py
    * general variables and paths for the RubiCube framework. 
* transient_lumonosity.py
    * methods for computing the transient neutrino luminosity
* transient_rates.py
    * methods for computing the Volumetric rate of transients
* Rubicube.py
    * Old version, ignore


#### User notebooks in RubiCube/notebooks
* `Realtime`: Everything related to realtime counterpart search for neutrinos
    * `Single alert explorer.ipynb`: Reads in the latest neutrino alert from the kafka stream and writes the ampel job file `Rubicube/ampel job files/TEST_ic_alert_ZTF_processor.yaml`. The job file can then be run with:
    ``` poetry run ampel job Rubicube/ampel\ job\ files/TEST_ic_alert_ZTF_processor.yaml -config ampel_conf.yaml -secrets secrets.yaml -reset-db```

* `Rubin Alerts`: Adapting things to rubin alerts
    * `generate_IC_map.ipynb`: notebook where you can set a RA, DEC, and time, and you output an IC alert
    * `ic_map_generator.py`: methods for map generating

* `bayesian analysis`: Here we are building the simulation to test the bayesian framework. It uses functions from another simulation code [TransientSimulation](https://github.com/krissylla/TransientSimulation), so this needs to be downloaded first to be able to test things here.  
