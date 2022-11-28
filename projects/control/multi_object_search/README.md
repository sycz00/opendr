# OpenDR mobile manipulation demo
<div align="left">
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" height="20">
  </a>
</div>

Live demo of mobile manipulation using the [OpenDR toolkit](https://opendr.eu).


## Set-up
Follow the setup described for the [multi_object_search tool](/docs/reference/multi_object_search.md). 

## Running the example
Mobile manipulation tasks in the analytical environment can be run as follows:
```bash
python multi_object_search_demo.py
```

They demo either executes a training or evaluation instance. By default it will evaluate all evaluation-scenes for 75 episodes sequentially.
The following will list the most important flags in 
```bash
  best_defaults.yaml
```
Uncomment or comment (#) the desired robot by either using the LoCoBot or Fetch. The training can be specified by `evaluation = false`(true by default).


## Acknowledgement
This work has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 871449 (OpenDR). This publication reflects the authors’ views only. The European Commission is not responsible for any use that may be made of the information it contains.
