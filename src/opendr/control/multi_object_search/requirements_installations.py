from igibson.utils.assets_utils import download_assets,download_ig_dataset,download_demo_data
from opendr.engine.constants import OPENDR_SERVER_URL
from pathlib import Path
import shutil
import os
TEMP_SAVE_DIR = Path(__file__).parent / "multi_object_search_tmp"
from opendr.engine.constants import OPENDR_SERVER_URL
from urllib.request import urlretrieve
import igibson
"""
This file is specifically for downloading the iGibson dataset (without the key) and the inflated maps
"""
class multi_object_search_requirements():
	def __init__(self):
		self.temp_path = TEMP_SAVE_DIR

	def download_dataset(self, path=None, url=OPENDR_SERVER_URL + "control/multi_object_search/"):
		path = self.temp_path
		print("-----> Start downloading iGibson assets")
		download_ig_dataset()
		print("-----> Successfully downloaded iGibson Dataset")
		inflated_files = ["Beechwood_0_int","Beechwood_1_int","Benevolence_0_int","Benevolence_1_int","Benevolence_2_int","Ihlen_0_int",\
		"Ihlen_1_int","Merom_0_int","Merom_1_int","Pomaria_0_int","Pomaria_1_int","Pomaria_2_int","Rs_int","Wainscott_0_int","Wainscott_1_int"]
		print("-----> Start Download iGibson prerequisits")
		file_destinations = []
		for infl_map in inflated_files:
			filename = f"inflated_maps/scenes/{infl_map}/layout/floor_trav_no_obj_0.png"
			file_destination = Path(path) / filename
			file_destinations.append(file_destination)
			if not file_destination.exists():
				file_destination.parent.mkdir(parents=True, exist_ok=True)
				url_download = os.path.join(url, filename)
				urlretrieve(url=url_download, filename=file_destination)
			#copy all inflated maps to the corresponding iGibson folders
			all_subdirs = [scen_n for scen_n in os.listdir(Path(path) / f"inflated_maps/scenes/")]
			for scene_name in all_subdirs:
				inflated_src = Path(path) / f"inflated_maps/scenes/" / scene_name/"layout/floor_trav_no_obj_0.png"
				dst = Path(igibson.ig_dataset_path) / f"scenes/{scene_name}/layout/"
				shutil.copy(inflated_src, dst)
				graph_file = dst / "floor_trav_0_py38.p"
				#remove Graph file for corresponding traversability map
				if graph_file.exists():
					os.remove(graph_file)


		return file_destinations


if __name__ == "__main__":
	download = multi_object_search_requirements()
	download.download_dataset()