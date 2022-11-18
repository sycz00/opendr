from test_multi_object_search import MultiObjectSearchTest


def main():
	stuff = MultiObjectSearchTest()
	stuff.setUpClass()

	#stuff.test_ckpt_download()
	print("CHEEEEEEEEEEEEEEEEEEEEEEEECK")
	input()
	stuff.test_prereqs_download()
	print("Test Complete")
	#stuff.test_eval()
	#stuff.test_eval_pretrained()
	
	


main()