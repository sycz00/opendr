from test_multi_object_search import MultiObjectSearchTest
from pathlib import Path
TEMP_SAVE_DIR = Path(__file__).parent / "multi_object_search_tmp/"
def main():

	if not TEMP_SAVE_DIR.exists():
		TEMP_SAVE_DIR.mkdir(parents=True, exist_ok=True)
	print(str(TEMP_SAVE_DIR))
	#stuff = MultiObjectSearchTest()
	#stuff.setUpClass()
	#stuff.env.reset()
	#print("SUCCESSFULLY !!!!")
	#input()
	#stuff.test_fit()
	print("READY")
	#stuff.test_eval()
	#stuff.learner.fit()

	#stuff.test_ckpt_download()
	#print("CHEEEEEEEEEEEEEEEEEEEEEEEECK")
	#input()
	#stuff.test_prereqs_download()
	#print("Test Complete")
	#stuff.test_eval()
	#stuff.test_eval_pretrained()
	
	


main()